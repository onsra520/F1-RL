"""Manifest, historical provenance and run-verifier CLI extraction regressions.

All data created by these tests is synthetic, local, and non-scientific.
"""
from pathlib import Path
import json
import shutil
import subprocess
import sys

import pytest

from src.validation.evidence_manifest_workflow import main as manifest_main
from src.validation.historical_source_audit_workflow import main as historical_main
from src.validation.run_evidence_workflow import main as run_main
from src.validation.wp7_historical_provenance import HistoricalSourceAudit

ROOT = Path(__file__).resolve().parents[2]


def test_new_manifest_roundtrip_rejects_overwrite_and_detects_tamper(tmp_path, capsys):
    source = tmp_path / "original"
    source.mkdir()
    (source / "test.txt").write_bytes(b"F1RL witness\n")
    destination = tmp_path / "restored"
    shutil.copytree(source, destination)
    manifest = tmp_path / "manifest.json"

    args = ["--source", str(source), "--manifest", str(manifest)]
    assert manifest_main(["create", *args]) == 0
    record = json.loads(manifest.read_text(encoding="utf-8"))
    assert record["algorithm"] == "sha256"
    assert record["entries"][0]["path"] == "test.txt"
    assert manifest_main(["verify", "--source", str(destination), "--manifest", str(manifest)]) == 0
    assert "RESTORE VERIFIED: 1 files, SHA256 exact" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        manifest_main(["create", *args])
    (destination / "test.txt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="differs"):
        manifest_main(["verify", "--source", str(destination), "--manifest", str(manifest)])


def test_duplicate_json_manifest_keys_are_rejected(tmp_path):
    source = tmp_path / "evidence"
    source.mkdir()
    (source / "witness").write_bytes(b"abc")
    manifest = tmp_path / "duplicate.json"
    manifest.write_text('{"schema_version":1,"schema_version":1,"algorithm":"sha256","entries":[]}',
                        encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate evidence JSON key"):
        manifest_main(["verify", "--source", str(source), "--manifest", str(manifest)])


def test_manifest_output_cannot_live_inside_scanned_tree(tmp_path):
    source = tmp_path / "tree"
    source.mkdir()
    (source / "witness").write_bytes(b"abc")
    with pytest.raises(ValueError, match="outside the scanned evidence tree"):
        manifest_main(["create", "--source", str(source), "--manifest", str(source / "manifest.json")])
    assert not (source / "manifest.json").exists()


def test_run_verifier_rejects_partial_review_pair(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        run_main([
            "--manifest", str(tmp_path / "missing.json"),
            "--expected-sha256", "0" * 64,
            "--review", str(tmp_path / "review.json"),
        ])
    assert exc.value.code == 2
    assert "--review and --trust-registry must be supplied together" in capsys.readouterr().err


def test_historical_audit_rejects_worktree_output(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        historical_main(ROOT, [
            "--archive-run", str(tmp_path / "archive"),
            "--output", str(ROOT / "docs" / "unsafe-witness.json"),
        ])
    assert exc.value.code == 2
    assert "outside the source worktree" in capsys.readouterr().err
    assert not (ROOT / "docs" / "unsafe-witness.json").exists()


def test_historical_audit_never_approves_scientific_work(tmp_path, monkeypatch):
    from src.validation import historical_source_audit_workflow as handler
    archive = tmp_path / "archive"
    archive.mkdir()
    result = HistoricalSourceAudit("0" * 40, True, 8, (), ())
    monkeypatch.setattr(handler, "audit_historical_manifest_file", lambda root, manifest: result)
    output = tmp_path / "audit" / "result.json"
    assert handler.main(ROOT, ["--archive-run", str(archive), "--output", str(output)]) == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["training_permitted"] is False
    assert document["heldout_permitted"] is False
    assert document["scientific_acceptance_permitted"] is False
    assert document["archive_runs"][0]["training_runtime_provenance_verified"] is False
    with pytest.raises(SystemExit) as exc:
        handler.main(ROOT, ["--archive-run", str(archive), "--output", str(output)])
    assert exc.value.code == 2


@pytest.mark.parametrize("module,group,action", [
    ("scripts.evidence.evidence_manifest", "evidence", "evidence-manifest"),
    ("scripts.evidence.verify_run", "evidence", "verify-run"),
    ("scripts.evidence.historical_source_audit", "evidence", "historical-source-audit"),
])
def test_grouped_and_stable_evidence_help_match_without_executing(module, group, action):
    old = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    modern = subprocess.run(
        [sys.executable, "-m", "scripts.workflows", group, action, "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert old.returncode == modern.returncode == 0, (old.stderr, modern.stderr)
    assert old.stdout == modern.stdout
