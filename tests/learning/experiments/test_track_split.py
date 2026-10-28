import json
from pathlib import Path

import pytest

from src.learning.experiments import (
    TrackExperimentConfig,
    resolve_track_split,
)
from src.track_assets import load_track_asset


def _track(track_id: str, *, spawn_x: float = 0.0, surface_id: str | None = None) -> dict:
    surfaces = []
    if surface_id is not None:
        surfaces.append(
            {
                "id": surface_id,
                "origin": [0.0, 0.0, 0.0],
                "forward": [0.0, 0.0, 1.0],
                "normal": [0.0, 1.0, 0.0],
                "width": 10.0,
                "length": 20.0,
            }
        )
    return {
        "schema_version": 1,
        "track_id": track_id,
        "spawn": {"position": [spawn_x, 0.0, 0.0], "orientation": [1.0, 0.0, 0.0, 0.0]},
        "surfaces": surfaces,
        "obstacles": [],
        "boost_pads": [],
        "slow_zones": [],
        "checkpoints": [],
        "finish_lines": [],
        "score_pickups": [],
    }


def _entry(
    root: Path,
    rel: str,
    track_id: str,
    *,
    spawn_x: float = 0.0,
    surface_id: str | None = None,
) -> dict:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_track(track_id, spawn_x=spawn_x, surface_id=surface_id)),
        encoding="utf-8",
    )
    asset = load_track_asset(path)
    return {
        "track_id": track_id,
        "path": rel,
        "world_sha256": asset.world_sha256,
        "content_sha256": asset.content_sha256,
    }


def _manifest(root: Path, entries: list[dict]) -> Path:
    path = root / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "tracks": entries}), encoding="utf-8")
    return path


def test_resolve_track_split_preserves_declared_order_and_ignores_directory_roles(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path, "evaluation/train-a.json", "train-a", surface_id="surface-a"),
        _entry(tmp_path, "training/train-b.json", "train-b", surface_id="surface-b"),
        _entry(tmp_path, "training/heldout-a.json", "heldout-a", surface_id="surface-held"),
        _entry(tmp_path, "demo/extra.json", "extra", surface_id="surface-extra"),
    ]
    config = TrackExperimentConfig(
        _manifest(tmp_path, entries),
        ("train-b", "train-a"),
        ("heldout-a",),
    )

    split = resolve_track_split(config)

    assert tuple(entry.track_id for entry in split.training) == ("train-b", "train-a")
    assert tuple(entry.track_id for entry in split.heldout) == ("heldout-a",)
    assert split.training[1].path == "evaluation/train-a.json"
    assert split.heldout[0].path == "training/heldout-a.json"
    assert "extra" not in {
        entry.track_id for entry in (*split.training, *split.heldout)
    }


def test_resolve_track_split_rejects_declared_id_missing_from_manifest(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path, "training/train-a.json", "train-a", surface_id="surface-a"),
        _entry(tmp_path, "evaluation/heldout-a.json", "heldout-a", surface_id="surface-held"),
    ]
    config = TrackExperimentConfig(
        _manifest(tmp_path, entries),
        ("train-a", "missing"),
        ("heldout-a",),
    )

    with pytest.raises(ValueError, match="missing|manifest"):
        resolve_track_split(config)


@pytest.mark.parametrize("duplicate_partition", ["training", "heldout"])
def test_resolve_track_split_rejects_duplicate_world_hash_within_partition(
    tmp_path: Path,
    duplicate_partition: str,
) -> None:
    train_entries = [
        _entry(tmp_path, "training/train-a.json", "train-a", surface_id=None),
        _entry(tmp_path, "training/train-b.json", "train-b", spawn_x=5.0, surface_id=None),
    ]
    heldout_entries = [
        _entry(tmp_path, "evaluation/heldout-a.json", "heldout-a", surface_id="held-surface"),
        _entry(tmp_path, "evaluation/heldout-b.json", "heldout-b", spawn_x=8.0, surface_id="held-surface"),
    ]

    if duplicate_partition == "training":
        entries = train_entries + [heldout_entries[0]]
        training_ids = ("train-a", "train-b")
        heldout_ids = ("heldout-a",)
    else:
        entries = [train_entries[0]] + heldout_entries
        training_ids = ("train-a",)
        heldout_ids = ("heldout-a", "heldout-b")

    config = TrackExperimentConfig(_manifest(tmp_path, entries), training_ids, heldout_ids)
    with pytest.raises(ValueError, match="world_sha256|geometry|world hash"):
        resolve_track_split(config)


def test_resolve_track_split_rejects_same_world_under_different_ids_and_spawns(tmp_path: Path) -> None:
    train = _entry(tmp_path, "training/train-a.json", "train-a", spawn_x=0.0)
    heldout = _entry(tmp_path, "evaluation/heldout-a.json", "heldout-a", spawn_x=10.0)
    assert train["world_sha256"] == heldout["world_sha256"]
    assert train["content_sha256"] != heldout["content_sha256"]

    config = TrackExperimentConfig(_manifest(tmp_path, [train, heldout]), ("train-a",), ("heldout-a",))
    with pytest.raises(ValueError, match="world_sha256|geometry|world hash"):
        resolve_track_split(config)
