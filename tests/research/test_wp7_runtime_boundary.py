"""Runtime capture is separate from WP7 protocol construction and admission."""
import ast
from pathlib import Path

import pytest

from src.validation.wp7_runtime import hash_model_artifacts


ROOT = Path(__file__).resolve().parents[2]


def test_model_artifact_hashing_keeps_zip_identity_and_exact_bytes(tmp_path):
    artifact = tmp_path / "checkpoint.zip"
    artifact.write_bytes(b"frozen model fixture")

    result = hash_model_artifacts({"policy-seed0": artifact})

    import hashlib

    assert result == {
        "policy-seed0": {
            "sha256": hashlib.sha256(b"frozen model fixture").hexdigest(),
            "size_bytes": len(b"frozen model fixture"),
        }
    }


@pytest.mark.parametrize("artifacts", [None, [], {" ": "x.zip"}])
def test_model_artifact_hashing_rejects_invalid_identity_map(artifacts):
    with pytest.raises(ValueError, match="artifact identities"):
        hash_model_artifacts(artifacts)


def test_model_artifact_hashing_rejects_non_zip_and_missing_files(tmp_path):
    text_file = tmp_path / "model.json"
    text_file.write_text("{}", encoding="utf-8")
    missing = tmp_path / "missing.zip"

    for artifact in (text_file, missing):
        with pytest.raises(ValueError, match="existing model/checkpoint ZIP"):
            hash_model_artifacts({"candidate": artifact})


def test_admission_imports_the_public_runtime_capture_boundary():
    tree = ast.parse((ROOT / "src/validation/wp7_admission.py").read_text(encoding="utf-8"))
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module in {"wp7_protocol", "wp7_runtime"}
        for alias in node.names
    }

    assert "capture_runtime_fingerprint" in imported_names
    assert "_runtime" not in imported_names
