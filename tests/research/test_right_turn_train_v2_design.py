"""Versioned Right Turn TRAIN v2: mirrored pair, never mutate historic TRAIN-4."""
from hashlib import sha256
from math import cos, pi, sin
import json
from pathlib import Path

import pytest

from f1rl.learning.pilot_preflight import TRAIN_IDS
from f1rl.sandbox.center_trajectory import CenterSegment
from f1rl.sandbox.directed_center_crossing import directed_center_gate_crossing
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.track_assets import load_track_asset, load_track_manifest
from scripts.wp7_train5_handoff_preflight import audit as audit_source_lock

ROOT = Path(__file__).resolve().parents[2]
RIGHT = ROOT / "assets/tracks/training/right-turn-train-v2.json"
LEFT = ROOT / "assets/tracks/training/left-turn-train-v2.json"
MANIFEST = ROOT / "assets/tracks/manifest-five-track-paired-v2.json"
CANDIDATE = ROOT / "configs/recovery/wp7-c2-five-track-paired-v2.yml"
PAIRED_IDS = (
    "straight-sprint-v1", "slalom-v1", "right-turn-train-v2",
    "ramp-v1", "left-turn-train-v2",
)
FROZEN_RIGHT_SHA256 = "a302f5d1d7f08d2b51ae963a5e695d8600d52a0ebf2d1f212ca0a4fbf90e303e"


def test_right_v2_is_independent_versioned_asset_and_valid_navigation():
    asset = load_track_asset(RIGHT)
    assert asset.schema_version == 2
    assert asset.track_id == "right-turn-train-v2"
    doc = json.loads(RIGHT.read_text(encoding="utf-8"))
    assert doc["spawn"]["position"] == [-3, 0, 5]
    assert [(x["origin"], x["width"], x["length"]) for x in doc["surfaces"]] == [
        ([-3, 0, 44], 16, 88), ([43, 0, 75], 16, 92)
    ]
    assert len(doc["navigation"]["points"]) == 11
    assert doc["navigation"]["checkpoint_point_indices"] == [1, 8]
    assert doc["navigation"]["surface_ids"] == ["road-entry"] * 6 + ["road-exit"] * 4
    for i, point in enumerate(doc["navigation"]["points"][3:8]):
        angle = i*pi/8
        assert point[0] == pytest.approx(7 - 10*cos(angle), abs=2e-6)
        assert point[2] == pytest.approx(65 + 10*sin(angle), abs=2e-6)
    assert asset.world.surface_at(asset.spawn.position, 0) is not None


def test_turn_v2_paired_geometry_mirror_and_directed_gates():
    right = json.loads(RIGHT.read_text(encoding="utf-8"))
    left = json.loads(LEFT.read_text(encoding="utf-8"))
    assert right["track_id"] != left["track_id"]
    for a,b in zip(right["surfaces"],left["surfaces"],strict=True):
        assert a["origin"] == [-b["origin"][0], b["origin"][1],b["origin"][2]]
        assert a["forward"] == [-b["forward"][0],b["forward"][1],b["forward"][2]]
        assert (a["width"],a["length"])==(b["width"],b["length"])
    for a,b in zip(right["navigation"]["points"],left["navigation"]["points"],strict=True):
        assert a==[-b[0],b[1],b[2]]
    for name in ("checkpoints","finish_lines"):
        for a,b in zip(right[name],left[name],strict=True):
            ca=a["volume"]["center"]
            cb=b["volume"]["center"]
            assert ca == [-cb[0],cb[1],cb[2]]
    assert right["navigation"]["checkpoint_point_indices"] == left["navigation"]["checkpoint_point_indices"]


@pytest.mark.parametrize("gate_id,expected_center,expected_direction",[
    ("checkpoint-0", Vec3(-3,0,38),Vec3(0,0,1)),
    ("checkpoint-1", Vec3(24,0,75),Vec3(1,0,0)),
    ("finish", Vec3(80,0,75),Vec3(1,0,0)),
])
def test_actual_v2_gate_directed_forward_only(gate_id,expected_center,expected_direction):
    asset = load_track_asset(RIGHT)
    gates = [*asset.world.checkpoints,*asset.world.finish_lines]
    gate = next(g for g in gates if g.id == gate_id)
    volume=gate.volume
    assert volume.center == expected_center
    assert volume.orientation.rotate(Vec3(0,0,1)).dot(expected_direction) == pytest.approx(1)
    start=expected_center-expected_direction*3+Vec3(0,.5,0)
    end=expected_center+expected_direction*3+Vec3(0,.5,0)
    assert directed_center_gate_crossing(start,end,volume) is not None
    assert CenterSegment(0,1,start,end).directed_crossings(volume) == pytest.approx((.5,))
    assert directed_center_gate_crossing(end,start,volume) is None
    assert CenterSegment(0,1,end,start).directed_crossings(volume) == ()


def test_paired_manifest_preserves_frozen_metadata_and_has_unique_hashes():
    old=json.loads((ROOT/"assets/tracks/manifest.json").read_text(encoding="utf-8"))
    new=json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert new["schema_version"] == 1 and len(new["tracks"]) == 7
    assert new["tracks"][:2] == old["tracks"][:2]
    assert new["tracks"][3] == old["tracks"][3]
    assert new["tracks"][5:] == old["tracks"][4:]
    assert tuple(e["track_id"] for e in new["tracks"][:5]) == PAIRED_IDS
    known_world={e["world_sha256"] for e in old["tracks"]}
    known_content={e["content_sha256"] for e in old["tracks"]}
    for idx,path in ((2,RIGHT),(4,LEFT)):
        e=new["tracks"][idx]
        asset=load_track_asset(path)
        assert (e["world_sha256"],e["content_sha256"]) == (asset.world_sha256,asset.content_sha256)
        assert asset.world_sha256 not in known_world
        assert asset.content_sha256 not in known_content
    assert new["tracks"][2]["world_sha256"]!=new["tracks"][4]["world_sha256"]
    assert new["tracks"][2]["content_sha256"]!=new["tracks"][4]["content_sha256"]
    verified=load_track_manifest(MANIFEST,validate_asset_ids=PAIRED_IDS)
    assert tuple(x.track_id for x in verified.tracks[:5])==PAIRED_IDS


def test_paired_training_factory_uses_five_train_assets_without_reading_heldout(monkeypatch):
    from f1rl.learning.reconstruction import RecoveryCandidate,build_training_scenarios
    import f1rl.learning.reconstruction as r
    candidate=RecoveryCandidate.from_yaml(CANDIDATE)
    experiment=candidate.experiment
    assert experiment.metadata.id == "ppo-generalization-five-track-paired-v2"
    assert experiment.tracks.training_ids==PAIRED_IDS
    assert experiment.tracks.heldout_ids==("left-turn-v1","offset-slalom-v1")
    assert experiment.tracks.manifest.resolve() == MANIFEST.resolve()
    opened=[]
    actual=r.load_track_asset
    def checked(path):
        assert "/evaluation/" not in Path(path).as_posix()
        opened.append(Path(path).name)
        return actual(path)
    monkeypatch.setattr(r,"load_track_asset",checked)
    scenarios=build_training_scenarios(candidate)
    assert tuple(x.track_id for x in scenarios.scenarios)==PAIRED_IDS
    assert len(opened)==5 and "right-turn-train-v2.json" in opened
    assert "left-turn-train-v2.json" in opened
    assert all(x.factory.track is not None for x in scenarios.scenarios)


def test_frozen_right_turn_remains_unchanged_and_no_training_authorization():
    assert TRAIN_IDS == ("straight-sprint-v1","slalom-v1","right-turn-v1","ramp-v1")
    path=ROOT/"assets/tracks/training/right-turn-v1.json"
    assert audit_source_lock(ROOT)["source_file_git_blob_hashes_verified"] == 44
    registered = json.loads((
        ROOT / "configs/research/wp7-train5-paired-v2-registration.json"
    ).read_text(encoding="utf-8"))["source_sha256"]
    assert registered["assets/tracks/training/right-turn-v1.json"] == FROZEN_RIGHT_SHA256
    assert not (RIGHT.parent/"right-turn-train-v2.json").read_text().count("training_authorized")
