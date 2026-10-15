"""Synthetic git-history fixtures proving read-only WP7 source reconstruction.

This test suite never touches real archived PPO evidence nor permits TRAIN/
HELD-OUT execution. It also tests schema and failure paths on Windows/Linux.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess

import pytest

from f1rl.research.wp7_historical_provenance import (
    ARCHIVED_SOURCE_PATHS, audit_archived_git_sources,
    audit_historical_manifest_file,
)


def cmd(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout.decode("ascii").strip()


@pytest.fixture()
def historical_repo(tmp_path: Path):
    cmd(tmp_path, "init")
    cmd(tmp_path, "config", "user.email", "wp7-test@example.invalid")
    cmd(tmp_path, "config", "user.name", "WP7 test")
    entries = []
    for path in ARCHIVED_SOURCE_PATHS:
        payload = ("historical synthetic source: " + path + "\n").encode("utf-8")
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        entries.append({"path": path, "sha256": sha256(payload).hexdigest()})
    cmd(tmp_path, "add", "--", *ARCHIVED_SOURCE_PATHS)
    cmd(tmp_path, "commit", "-m", "synthetic historical source")
    commit = cmd(tmp_path, "rev-parse", "HEAD")
    manifest = {
        "git_commit": commit,
        "source_fingerprints": entries,
        "heldout_rollouts": 0,
        "scientific_evidence": False,
    }
    return tmp_path, manifest


def test_original_training_git_bytes_match_without_scientific_admission(historical_repo):
    root, manifest = historical_repo
    result = audit_archived_git_sources(root, manifest)
    assert result.archived_source_verified
    assert result.archived_fingerprint_count == 8
    assert result.current_worktree_drifts == ()
    assert result.failures == ()
    assert not result.training_runtime_provenance_verified
    assert not result.scientific_acceptance_permitted


def test_current_worktree_config_change_does_not_rewrite_archived_run(historical_repo):
    root, manifest = historical_repo
    config = root / ARCHIVED_SOURCE_PATHS[0]
    config.write_text("subsequent simulator configuration\n", encoding="utf-8")
    result = audit_archived_git_sources(root, manifest)
    assert result.archived_source_verified
    assert result.current_worktree_drifts == (ARCHIVED_SOURCE_PATHS[0],)
    assert not result.scientific_acceptance_permitted
    assert cmd(root, "show", f"{manifest['git_commit']}:{ARCHIVED_SOURCE_PATHS[0]}")


def test_forged_archived_fingerprint_is_rejected(historical_repo):
    root, manifest = historical_repo
    manifest["source_fingerprints"][1]["sha256"] = "0" * 64
    result = audit_archived_git_sources(root, manifest)
    assert not result.archived_source_verified
    assert any(ARCHIVED_SOURCE_PATHS[1] in reason for reason in result.failures)


def test_historical_commit_must_exist(historical_repo):
    root, manifest = historical_repo
    manifest["git_commit"] = "0" * 40
    result = audit_archived_git_sources(root, manifest)
    assert not result.archived_source_verified


@pytest.mark.parametrize("mode", ["missing", "duplicate", "traversal", "bad_sha"])
def test_source_inventory_and_digests_are_strict(historical_repo, mode):
    root, manifest = historical_repo
    if mode == "missing":
        manifest["source_fingerprints"].pop()
    elif mode == "duplicate":
        manifest["source_fingerprints"][1] = manifest["source_fingerprints"][0]
    elif mode == "traversal":
        manifest["source_fingerprints"][0]["path"] = "../.git/config"
    else:
        manifest["source_fingerprints"][0]["sha256"] = "NOT-SHA256"
    with pytest.raises(ValueError):
        audit_archived_git_sources(root, manifest)


def test_heldout_protected_regardless_of_historical_fingerprint(historical_repo):
    root, manifest = historical_repo
    manifest["heldout_rollouts"] = 1
    with pytest.raises(ValueError, match="sealed exploratory"):
        audit_archived_git_sources(root, manifest)


def test_manifest_read_does_not_modify_archive(historical_repo):
    root, manifest = historical_repo
    archive = root.parent / "archive-manifest.json"
    archive.write_text(json.dumps(manifest), encoding="utf-8")
    initial_digest = sha256(archive.read_bytes()).hexdigest()
    output = audit_historical_manifest_file(root, archive)
    assert output.archived_source_verified
    assert sha256(archive.read_bytes()).hexdigest() == initial_digest


def test_historical_windows_crlf_checkout_is_traced_not_silently_rehashed(historical_repo):
    root, manifest = historical_repo
    path = ARCHIVED_SOURCE_PATHS[2]
    original = (root / path).read_bytes()
    archived_windows_bytes = original.replace(b"\n", b"\r\n")
    row = next(item for item in manifest["source_fingerprints"] if item["path"] == path)
    row["sha256"] = sha256(archived_windows_bytes).hexdigest()
    result = audit_archived_git_sources(root, manifest)
    assert result.archived_source_verified
    assert (path, "git-lf-to-windows-crlf") in result.historical_eol_modes
    assert path in result.current_worktree_drifts
    assert not result.training_runtime_provenance_verified
