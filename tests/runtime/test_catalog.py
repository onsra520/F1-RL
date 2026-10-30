import hashlib
import json
from pathlib import Path

import pytest

from src.rl.pilot_preflight import TRAIN_IDS
from src.runtime.catalog import RuntimeCatalog

ROOT = Path(__file__).resolve().parents[2]
MODEL_MANIFEST = ROOT / "docs/research/evidence/wp3-issue51-frozen-final-models-20261028/WP3-FROZEN-MODELS-SHA256-MANIFEST.json"


def test_catalog_verifies_and_exposes_exact_frozen_models():
    catalog = RuntimeCatalog(ROOT)
    expected_ids = tuple(f"{candidate}-seed{seed}" for candidate in "ABCD" for seed in range(3))

    assert tuple(model.model_id for model in catalog.models) == expected_ids
    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["heldout_access"] is False
    for model in manifest["models"]:
        path = ROOT / "docs/research/evidence/wp3-issue51-frozen-final-models-20261028" / model["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == model["sha256"]
        assert catalog.model(model["model_id"]).sha256 == model["sha256"]


def test_catalog_only_exposes_registered_train_tracks_and_rejects_unknown_ids():
    catalog = RuntimeCatalog(ROOT)

    assert tuple(track.track_id for track in catalog.tracks) == TRAIN_IDS
    with pytest.raises(ValueError, match="frozen model"):
        catalog.model("../models/other.zip")
    with pytest.raises(ValueError, match="TRAIN track"):
        catalog.create_environment("heldout-track", seed=0)


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
def test_catalog_rejects_invalid_runtime_seeds(seed):
    catalog = RuntimeCatalog(ROOT)

    with pytest.raises(ValueError, match="seed"):
        catalog.create_environment(TRAIN_IDS[0], seed=seed)
