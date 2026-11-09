"""Build and check an unsigned, non-authorizing TRAIN-5 review manifest."""
from hashlib import sha256
import json
from pathlib import Path

from f1rl.research.wp7_train5_v4_admission import load_v4_protocol, validate_v4_protocol
from f1rl.research.wp7_train5_v4_checkpoint import FROZEN_CHECKPOINTS
from f1rl.research.wp7_protocol import _unique


REVIEW_SOURCES = (
    "configs/research/wp7-train5-internal-owner-key-v1.json",
    "scripts/wp7_train5_batch.py",
    "scripts/wp7_train5_v4_development.py",
    "scripts/wp7_train5_v4_factory_preview.py",
    "scripts/wp7_train5_execution_authorization.py",
    "scripts/wp7_train5_v4_review_manifest.py",
    "tests/research/test_wp7_train5_v4_factory_preview.py",
    "tests/research/test_wp7_train5_execution_authorization.py",
    "tests/research/test_wp7_train5_authorized_integration.py",
    "tests/research/test_wp7_train5_batch.py",
    "tests/research/test_wp7_train5_development.py",
    "tests/research/test_wp7_train5_v4_review_manifest.py",
    ".github/workflows/handoff-train5-engineering-preflight.yml",
)
MANIFEST = "configs/research/wp7-train5-v4-run-manifest-review-draft.json"


def _digest(root, relative, *, binary=False):
    root = Path(root).resolve(strict=True)
    path = root / relative
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError("review source must stay inside the checkout")
    data = path.read_bytes()
    return sha256(data if binary else data.replace(b"\r\n", b"\n")).hexdigest()


def build_review_manifest(root):
    root = Path(root).resolve(strict=True)
    protocol, protocol_hash = load_v4_protocol(root)
    validate_v4_protocol(root, protocol)
    checkpoints = {}
    for identity, expected in FROZEN_CHECKPOINTS.items():
        path = f"models/{identity}/model-final.zip"
        actual = _digest(root, path, binary=True)
        if actual != expected:
            raise ValueError(f"frozen checkpoint differs: {identity}")
        checkpoints[identity] = {"path": path, "sha256": actual}
    return {
        "schema": "f1rl_train5_v4_run_manifest_review_draft_v1",
        "status": "DRAFT_NOT_AUTHORIZED",
        "hash_convention": "sha256; text normalized CRLF to LF; model zip raw bytes",
        "protocol": {"path": "configs/research/wp7-train5-v4-simulation-protocol-draft.json", "sha256": protocol_hash},
        "review_source_bindings": {path: _digest(root, path) for path in REVIEW_SOURCES},
        "claim_scope": protocol["claim_scope"],
        "track_ids": protocol["track_ids"],
        "runs": protocol["runs"],
        "frozen_checkpoints": checkpoints,
        "budget": protocol["budget"],
        "reset": protocol["reset"],
        "development": protocol["development"],
        "permissions": protocol["permissions"],
        "review": protocol["review"],
        "limitations": ["source lock does not authenticate reviewer or owner",
                        "factory preview is not connected to an admitting runner",
                        "seed history and compatibility gates remain open",
                        "training-specific execution permit is absent"],
    }


def validate_review_manifest(root, manifest):
    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)

    try:
        matches = canonical(manifest) == canonical(build_review_manifest(root))
    except (TypeError, ValueError) as exc:
        raise ValueError("review manifest differs from verified sources or draft protocol") from exc
    if not matches:
        raise ValueError("review manifest differs from verified sources or draft protocol")


def load_review_manifest(root):
    root = Path(root).resolve(strict=True)
    path = root / MANIFEST
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError("review manifest must stay inside the checkout")
    data = path.read_bytes().replace(b"\r\n", b"\n")
    manifest = json.loads(data, object_pairs_hook=_unique)
    validate_review_manifest(root, manifest)
    return manifest, sha256(data).hexdigest()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(build_review_manifest(root), indent=2, ensure_ascii=True))
