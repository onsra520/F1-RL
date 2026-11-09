"""Portable source lock and admission denials; no PPO, HELD-OUT or training."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from scripts import wp7_train5_handoff_preflight as h

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / h.REG


def test_source_lock_and_training_gates_are_not_bypassed():
    receipt = h.audit(ROOT)
    assert receipt["result"] == "PASS_SOURCE_LOCK_NO_TRAINING"
    assert receipt["source_file_git_blob_hashes_verified"] == 44
    assert (
        receipt["registered_windows_sha256_direct_matches"]
        + receipt["registered_windows_sha256_via_pinned_eol_bridge"]
        == 44
    )
    assert receipt["external_witness_status"] == "NOT_VERIFIED_EXTERNAL_ROOT_NOT_PROVIDED"
    assert receipt["external_witness_count"] == 0
    assert receipt["proposed_seed_count"] == 18
    assert receipt["seed_global_status"] == "PROPOSED_UNVERIFIED_GLOBAL"
    assert receipt["training_permission"] is False
    assert receipt["smoke_permission"] is False
    assert receipt["heldout_permission"] is False
    assert receipt["scientific_release_permission"] is False


def test_linux_lf_checkout_of_windows_crlf_registered_source_is_accepted(monkeypatch):
    original = Path.read_bytes
    asset_name = "assets/tracks/manifest.json"
    asset = ROOT / asset_name
    git_blob = subprocess.run(
        ["git", "-C", str(ROOT), "show", "HEAD:" + asset_name],
        check=True, capture_output=True,
    ).stdout
    bridge = json.loads((
        ROOT / "configs/research/wp7-train5-paired-v2-git-blob-lock.json"
    ).read_text(encoding="utf8"))["source_blobs"][asset_name]
    assert bridge["relation"] == "CRLF_to_LF_only"
    assert bridge["registered_windows_worktree_sha256"] != bridge["git_blob_sha256"]
    def bytes_override(path):
        if Path(path) == asset:
            return git_blob
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", bytes_override)
    receipt = h.audit(ROOT)
    assert receipt["source_file_git_blob_hashes_verified"] == 44
    assert receipt["registered_windows_sha256_via_pinned_eol_bridge"] >= 1


def test_portable_source_binding_rejects_non_eol_drift(monkeypatch):
    original = Path.read_bytes
    source = ROOT / "f1rl/research/wp7_track_feasibility.py"
    def bytes_override(path):
        contents = original(path)
        return contents + b"# semantic source drift\n" if Path(path) == source else contents
    monkeypatch.setattr(Path, "read_bytes", bytes_override)
    with pytest.raises(h.HandoffPreflightError, match="working source differs from anchored Git blob"):
        h.audit(ROOT)


@pytest.mark.parametrize("field,value", [
    ("status", "approved"),
    ("training_permission", True),
    ("smoke_permission", True),
    ("heldout_permission", True),
    ("owner_approval", "fake"),
    ("global_seed_registry_receipt", "fake"),
    ("seed_audit_status", "GLOBAL_VERIFIED"),
])
def test_forged_approval_fails_before_source_or_geometry_reads(monkeypatch, field, value):
    original = Path.read_text
    reg = json.loads(original(REG))
    reg[field] = value
    def text_override(path, *args, **kwargs):
        if Path(path) == REG:
            return json.dumps(reg)
        return original(path, *args, **kwargs)
    def blocked_bytes(path):
        pytest.fail(f"preflight accessed source before approval denial: {path}")
    monkeypatch.setattr(Path, "read_text", text_override)
    monkeypatch.setattr(Path, "read_bytes", blocked_bytes)
    with pytest.raises(h.HandoffPreflightError):
        h.audit(ROOT)


def test_draft_source_roster_omits_heldout_geometry():
    reg = json.loads(REG.read_text(encoding="utf8"))
    assert len(reg["source_sha256"]) == 44
    assert all("evaluation/" not in path for path in reg["source_sha256"])
    assert all("heldout" not in path.lower() for path in reg["source_sha256"])
    assert reg["heldout_ids"] == list(h.HELDOUT)


def test_missing_external_witnesses_stay_fail_closed(tmp_path):
    with pytest.raises(h.HandoffPreflightError):
        h.audit(ROOT, tmp_path)


def test_locked_git_blob_receipt_matches_registered_windows_digest():
    bridge = json.loads((
        ROOT / "configs/research/wp7-train5-paired-v2-git-blob-lock.json"
    ).read_text(encoding="utf8"))
    reg = json.loads(REG.read_text(encoding="utf8"))
    assert set(bridge["source_blobs"]) == set(reg["source_sha256"])
    assert len(bridge["source_blobs"]) == 44
    assert all(
        bridge["source_blobs"][name]["registered_windows_worktree_sha256"] == digest
        for name, digest in reg["source_sha256"].items()
    )
