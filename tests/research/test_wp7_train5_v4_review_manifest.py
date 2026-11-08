from copy import deepcopy
from pathlib import Path
import json

import pytest

from scripts.wp7_train5_v4_review_manifest import MANIFEST, build_review_manifest, validate_review_manifest


ROOT = Path(__file__).resolve().parents[2]


def test_saved_manifest_matches_sources_and_grants_no_permission():
    manifest = json.loads((ROOT / MANIFEST).read_text(encoding="utf8"))
    validate_review_manifest(ROOT, manifest)
    assert len(manifest["track_ids"]) == 5
    assert len(manifest["runs"]) == 6
    assert len(manifest["frozen_checkpoints"]) == 3
    assert all(value is False for value in manifest["permissions"].values())


@pytest.mark.parametrize("field", ["runs", "track_ids", "permissions", "review_source_bindings", "frozen_checkpoints"])
def test_manifest_tampering_is_rejected(field):
    manifest = deepcopy(build_review_manifest(ROOT))
    manifest[field] = None
    with pytest.raises(ValueError, match="manifest differs"):
        validate_review_manifest(ROOT, manifest)


@pytest.mark.parametrize("relative", ["scripts/wp7_train5_v4_factory_preview.py", "models/C-seed0/model-final.zip"])
def test_changed_or_missing_bound_material_is_rejected(monkeypatch, relative):
    manifest = build_review_manifest(ROOT)
    target = ROOT / relative
    original = Path.read_bytes

    def changed(path):
        if path == target:
            return original(path) + b"changed"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(ValueError):
        validate_review_manifest(ROOT, manifest)


def test_missing_checkpoint_is_rejected(monkeypatch):
    target = ROOT / "models/C-seed0/model-final.zip"
    original = Path.read_bytes

    def missing(path):
        if path == target:
            raise FileNotFoundError(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", missing)
    with pytest.raises((FileNotFoundError, ValueError)):
        build_review_manifest(ROOT)
