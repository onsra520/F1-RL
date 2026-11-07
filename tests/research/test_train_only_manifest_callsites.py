"""TRAIN-only callsites may verify only selected TRAIN asset bytes.

This is not a blanket weakening of manifest validation: default call still
requires all entries, and every selected TRAIN asset must pass its original
content/world SHA-256 checks. No HELD-OUT geometry is opened.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from f1rl.learning.experiments import ExperimentDefinition
from f1rl.learning import pilot_preflight as pilot
from f1rl.learning import provenance as runprov
from f1rl.track_assets import manifest as track_manifest

ROOT=Path(__file__).resolve().parents[2]
EXPERIMENT=ROOT/"configs/experiments/ppo-generalization-v1.yml"
TRAIN_IDS=pilot.TRAIN_IDS


def _watch_manifest(monkeypatch, source_module, expected):
    real=track_manifest.load_track_manifest
    calls=[]
    def strict_selected(path, *, validate_asset_ids=None):
        assert validate_asset_ids==expected, (
            "TRAIN-only callsite attempted all-manifest or wrong-track validation"
        )
        calls.append((Path(path),validate_asset_ids))
        return real(path,validate_asset_ids=validate_asset_ids)
    monkeypatch.setattr(source_module,"load_track_manifest",strict_selected)
    return calls


def test_pilot_preflight_does_not_require_heldout_geometry(monkeypatch):
    calls=_watch_manifest(monkeypatch,pilot,TRAIN_IDS)
    report=pilot.build_pilot_preflight(
        repository_root=ROOT,
        candidate_path=ROOT/"configs/recovery/wp7-c2-candidate.yml",
        runtime_config_path=ROOT/"configs/recovery/wp7-pilot-200k.yml",
    )
    assert report["status"]=="review_required"
    assert report["training_track_ids"]==list(TRAIN_IDS)
    assert calls and all(ids==TRAIN_IDS for _,ids in calls)
    assert report["heldout_rollouts"]==0


def test_provenance_validates_exact_selected_train_assets(monkeypatch):
    selected=tuple(reversed(TRAIN_IDS))
    calls=_watch_manifest(monkeypatch,runprov,selected)
    definition=ExperimentDefinition.from_yaml(EXPERIMENT)
    report=runprov.capture_run_provenance(
        repository_root=ROOT,
        experiment_path=EXPERIMENT,
        definition=definition,
        selected_track_ids=selected,
    )
    assert calls==[(definition.tracks.manifest,selected)]
    assert tuple(r.track_id for r in report.tracks)==selected
    assert all(len(r.world_sha256)==64 and len(r.content_sha256)==64 for r in report.tracks)


def test_train_only_manifest_still_rejects_bad_train_world_hash(tmp_path):
    # Copy the manifest *metadata* and original four TRAIN assets, never HELD-OUT.
    source_manifest=ROOT/"assets/tracks/manifest.json"
    raw=json.loads(source_manifest.read_text("utf8"))
    path=tmp_path/"assets/tracks/manifest.json"
    path.parent.mkdir(parents=True)
    for item in raw["tracks"]:
        if item["track_id"] in TRAIN_IDS:
            src=source_manifest.parent/item["path"]
            dest=path.parent/item["path"]
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(src.read_bytes())
    assert not (path.parent/"evaluation/left-turn-v1.json").exists()
    assert not (path.parent/"evaluation/offset-slalom-v1.json").exists()
    path.write_text(json.dumps(raw),encoding="utf8")
    assert len(track_manifest.load_track_manifest(path,validate_asset_ids=TRAIN_IDS).tracks)==6

    for item in raw["tracks"]:
        if item["track_id"]=="slalom-v1":
            item["world_sha256"]="0"*64
    path.write_text(json.dumps(raw),encoding="utf8")
    with pytest.raises(ValueError,match="world_sha256 mismatch"):
        track_manifest.load_track_manifest(path,validate_asset_ids=TRAIN_IDS)


def test_default_manifest_validation_still_requires_all_assets(tmp_path):
    source=ROOT/"assets/tracks/manifest.json"
    raw=json.loads(source.read_text("utf8"))
    path=tmp_path/"manifest.json"
    path.write_text(json.dumps(raw),encoding="utf8")
    for item in raw["tracks"]:
        if item["track_id"] in TRAIN_IDS:
            dst=path.parent/item["path"]
            dst.parent.mkdir(parents=True,exist_ok=True)
            dst.write_bytes((source.parent/item["path"]).read_bytes())
    assert len(track_manifest.load_track_manifest(path,validate_asset_ids=TRAIN_IDS).tracks)==6
    with pytest.raises(FileNotFoundError):
        track_manifest.load_track_manifest(path)
