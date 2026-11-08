from __future__ import annotations

from pathlib import Path
from hashlib import sha256
import json

import numpy as np
import pytest

import f1rl.local_demo.runtime.catalog as catalog_module
from f1rl.local_demo.runtime.catalog import RuntimeCatalog, _load_registered_train_manifest
from f1rl.research import wp7_train5_v2_protocol


TRAIN_IDS = (
    "straight-sprint-v1",
    "slalom-v1",
    "right-turn-train-v2",
    "ramp-v1",
    "left-turn-train-v2",
)


def test_runtime_catalog_builds_only_source_bound_train5_v4_factories(monkeypatch):
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if "/assets/tracks/evaluation/" in path.as_posix().lower():
            raise PermissionError("HELD-OUT geometry is forbidden in runtime catalog")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    root = Path(__file__).parents[2]
    catalog = RuntimeCatalog(root)
    assert tuple(item.model_id for item in catalog.models) == tuple(
        f"{cohort}-seed{seed}" for cohort in "ABCD" for seed in range(3)
    )
    assert tuple(item.track_id for item in catalog.tracks) == TRAIN_IDS
    assert all(item.split == "TRAIN" for item in catalog.tracks)

    for track_id in TRAIN_IDS:
        factory = catalog.track_factory(track_id)
        assert factory.physics_config.ground_motion_version == "spatial_road_edge_partial_lifecycle_v4"
        assert factory.race_config.version == "wp7_directed_lifecycle_crash_contract_v4"
        assert factory.action_contract.version == "steering-throttle-brake-v2"
        env = catalog.create_environment(track_id, seed=20261108, mode="auto")
        try:
            observation, _ = env.reset(seed=20261108)
            assert env.observation_space.contains(observation)
            assert env.action_space.shape == (3,)
        finally:
            env.close()


def test_registered_train_track_completes_with_deterministic_development_policy(monkeypatch):
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if "/assets/tracks/evaluation/" in path.as_posix().lower():
            raise PermissionError("HELD-OUT geometry is forbidden in runtime catalog")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    root = Path(__file__).parents[2]
    catalog = RuntimeCatalog(root)
    track_id = "straight-sprint-v1"
    checkpoint_count = len(catalog.track_factory(track_id).navigation.checkpoint_point_indices)
    env = catalog.create_environment(track_id, seed=20261108, mode="manual")
    try:
        observation, _ = env.reset(seed=20261108)
        assert env.observation_space.contains(observation)

        terminated = truncated = False
        info = {}
        for _ in range(1000):
            observation, _, terminated, truncated, info = env.step(
                np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
            )
            assert env.observation_space.contains(observation)
            if terminated or truncated:
                break

        assert terminated
        assert not truncated
        assert info["finished"] is True
        assert info["next_checkpoint_index"] == checkpoint_count
        assert info.get("failure_reason") is None
    finally:
        env.close()


def test_train_manifest_is_rejected_before_opening_heldout_geometry(tmp_path, monkeypatch):
    manifest_path = tmp_path / wp7_train5_v2_protocol.MANIFEST
    manifest_path.parent.mkdir(parents=True)
    manifest_bytes = json.dumps({
        "schema_version": 1,
        "tracks": [{"track_id": TRAIN_IDS[0], "path": "evaluation/fake.json"}],
    }).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    evaluation_root = tmp_path / "assets/tracks/evaluation"
    evaluation_root.mkdir()

    original_open = Path.open
    attempted_geometry_opens = []

    def guarded_open(path, *args, **kwargs):
        if path.resolve(strict=False).is_relative_to(evaluation_root.resolve(strict=False)):
            attempted_geometry_opens.append(path.as_posix())
            raise AssertionError("evaluation geometry was opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(
        catalog_module,
        "load_track_manifest",
        lambda *args, **kwargs: pytest.fail("manifest loader ran before path validation"),
    )
    with pytest.raises(ValueError, match="HELD-OUT"):
        _load_registered_train_manifest(
            tmp_path,
            {"source_sha256": {wp7_train5_v2_protocol.MANIFEST: sha256(manifest_bytes).hexdigest()}},
            {},
        )
    assert attempted_geometry_opens == []
