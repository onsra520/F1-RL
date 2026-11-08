"""TRAIN-only Left Turn v2 design proof; no PPO, no HELD-OUT geometry read."""
from hashlib import sha256
import json
from math import cos, pi, sin
from pathlib import Path

import pytest

from f1rl.track_assets import load_track_asset, load_track_manifest
from f1rl.learning.pilot_preflight import TRAIN_IDS
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.center_trajectory import CenterSegment
from f1rl.sandbox.directed_center_crossing import directed_center_gate_crossing
from scripts.wp7_train5_handoff_preflight import audit as audit_source_lock

ROOT = Path(__file__).resolve().parents[2]
V2_ID = "left-turn-train-v2"
V2_PATH = ROOT / "assets/tracks/training/left-turn-train-v2.json"
V2_MANIFEST = ROOT / "assets/tracks/manifest-five-track-v2.json"
OLD_MANIFEST = ROOT / "assets/tracks/manifest.json"
FROZEN_RIGHT = "a302f5d1d7f08d2b51ae963a5e695d8600d52a0ebf2d1f212ca0a4fbf90e303e"


def test_left_v2_authored_route_uses_new_corner_dimensions_and_arc():
    asset = load_track_asset(V2_PATH)
    assert asset.schema_version == 2 and asset.track_id == V2_ID
    data = json.loads(V2_PATH.read_text(encoding="utf-8"))
    assert len(data["surfaces"]) == 2
    entry, exit_ = data["surfaces"]
    assert (entry["width"], entry["length"], entry["origin"]) == (16, 88, [3, 0, 44])
    assert (exit_["width"], exit_["length"], exit_["origin"]) == (16, 92, [-43, 0, 75])
    assert data["spawn"]["position"] == [3, 0, 5]
    assert len(data["navigation"]["points"]) == 11
    assert data["navigation"]["checkpoint_point_indices"] == [1, 8]
    assert data["navigation"]["surface_ids"] == ["road-entry"] * 6 + ["road-exit"] * 4
    # New independent quarter-circle R=10 m; never copy/mirror frozen route points.
    arc = data["navigation"]["points"][3:8]
    for j, p in enumerate(arc):
        theta = j * pi / 8
        assert p[0] == pytest.approx(-7 + 10 * cos(theta), abs=2e-6)
        assert p[2] == pytest.approx(65 + 10 * sin(theta), abs=2e-6)
        assert p[1] == 0


def test_manifest_v2_is_separate_with_no_training_heldout_hash_collision():
    orig = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    new = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    assert new["schema_version"] == 1
    assert new["tracks"][:4] == orig["tracks"][:4]
    assert new["tracks"][5:] == orig["tracks"][4:]
    fifth = new["tracks"][4]
    assert fifth["track_id"] == V2_ID
    assert fifth["path"] == "training/left-turn-train-v2.json"
    assert fifth["world_sha256"] not in {row["world_sha256"] for row in orig["tracks"]}
    assert fifth["content_sha256"] not in {row["content_sha256"] for row in orig["tracks"]}
    asset = load_track_asset(V2_PATH)
    assert (fifth["world_sha256"], fifth["content_sha256"]) == (
        asset.world_sha256, asset.content_sha256
    )
    # validate_asset_ids loads only the five TRAIN files, not evaluation geometry.
    manifest = load_track_manifest(V2_MANIFEST, validate_asset_ids=(*TRAIN_IDS, V2_ID))
    assert tuple(x.track_id for x in manifest.tracks[:5]) == (*TRAIN_IDS, V2_ID)


def test_v2_directional_gates_align_with_route_and_order():
    asset = load_track_asset(V2_PATH)
    gates = (*asset.world.checkpoints, *asset.world.finish_lines)
    assert tuple(g.id for g in gates) == ("checkpoint-0", "checkpoint-1", "finish")
    assert tuple(g.order for g in asset.world.checkpoints) == (0, 1)
    expected_centers = (Vec3(3, 0, 38), Vec3(-24, 0, 75), Vec3(-80, 0, 75))
    for gate, center in zip(gates, expected_centers, strict=True):
        vol = gate.volume
        assert vol.center == center
        forward = vol.orientation.rotate(Vec3(0, 0, 1))
        tangent = Vec3(0, 0, 1) if gate.id == "checkpoint-0" else Vec3(-1, 0, 0)
        assert forward.dot(tangent) == pytest.approx(1, abs=1e-10)
        prev = center - tangent * 3 + Vec3(0, .5, 0)
        nxt = center + tangent * 3 + Vec3(0, .5, 0)
        assert directed_center_gate_crossing(prev, nxt, vol) is not None
        assert CenterSegment(0, 1, prev, nxt).directed_crossings(vol) == pytest.approx((.5,))
        assert directed_center_gate_crossing(nxt, prev, vol) is None
        assert CenterSegment(0, 1, nxt, prev).directed_crossings(vol) == ()


def test_legacy_four_train_assets_and_registry_remain_frozen():
    assert TRAIN_IDS == (
        "straight-sprint-v1", "slalom-v1", "right-turn-v1", "ramp-v1"
    )
    assert audit_source_lock(ROOT)["source_file_git_blob_hashes_verified"] == 44
    registered = json.loads((
        ROOT / "configs/research/wp7-train5-paired-v2-registration.json"
    ).read_text(encoding="utf-8"))["source_sha256"]
    assert registered["assets/tracks/training/right-turn-v1.json"] == FROZEN_RIGHT
    assert OLD_MANIFEST.read_bytes() != V2_MANIFEST.read_bytes()


def test_new_design_is_not_an_approved_training_or_physics_finish():
    # An asset/manifest geometry audit is NOT a real controller finish witness.
    data = json.loads(V2_PATH.read_text(encoding="utf-8"))
    assert "training_authorized" not in data
    assert "validated_physics_finish" not in data


def test_five_track_v2_candidate_prepares_five_train_factories_without_heldout_io(monkeypatch):
    from f1rl.learning.reconstruction import RecoveryCandidate, build_training_scenarios
    import f1rl.learning.reconstruction as reconstruction

    experiment = ROOT / "configs/experiments/ppo-generalization-five-track-v2.yml"
    candidate_file = ROOT / "configs/recovery/wp7-c2-five-track-v2.yml"
    assert experiment.is_file() and candidate_file.is_file()
    candidate = RecoveryCandidate.from_yaml(candidate_file)
    assert candidate.experiment.metadata.id == "ppo-generalization-five-track-v2"
    assert candidate.experiment.tracks.training_ids == (*TRAIN_IDS, V2_ID)
    assert candidate.experiment.tracks.heldout_ids == ("left-turn-v1", "offset-slalom-v1")
    assert candidate.experiment.tracks.manifest.resolve() == V2_MANIFEST.resolve()

    original = reconstruction.load_track_asset
    observed = []

    def guarded(path):
        pathname = Path(path).as_posix()
        assert "/evaluation/" not in pathname, "HELD-OUT geometry must never be opened"
        observed.append(pathname)
        return original(path)

    monkeypatch.setattr(reconstruction, "load_track_asset", guarded)
    scenarios = build_training_scenarios(candidate)
    assert tuple(row.track_id for row in scenarios.scenarios) == (*TRAIN_IDS, V2_ID)
    assert len(observed) == 5
    assert all("/training/" in name for name in observed)
    left_factory = scenarios.scenarios[-1].factory
    assert left_factory.initial_state.vehicle.position == Vec3(3, 0, 5)
    # Factory assembly is not an executed Physics v4 episode nor training admission.
    assert left_factory.track is not None
