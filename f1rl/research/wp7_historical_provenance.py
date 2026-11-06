"""Verify historical WP7 source bytes against archived Git commit, read-only.

Current worktrees often diverge from training-time source. Verify the
declared historical commit's *actual bytes* without mutating a historical
archive or trusting today's worktree to reproduce an old environment.

This is one narrow G4 source-provenance check. It is NOT training runtime
provenance, rollout robustness, restored off-device evidence, or approval.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import re
import subprocess

ARCHIVED_SOURCE_PATHS = (
    "configs/recovery/wp7-c2-candidate.yml",
    "configs/recovery/wp7-pilot-200k.yml",
    "configs/experiments/ppo-generalization-v1.yml",
    "assets/tracks/manifest.json",
    "assets/tracks/training/straight-sprint-v1.json",
    "assets/tracks/training/slalom-v1.json",
    "assets/tracks/training/right-turn-v1.json",
    "assets/tracks/training/ramp-v1.json",
)


@dataclass(frozen=True, slots=True)
class HistoricalSourceAudit:
    archived_git_commit: str
    archived_source_verified: bool
    archived_fingerprint_count: int
    current_worktree_drifts: tuple[str, ...]
    failures: tuple[str, ...]
    historical_eol_modes: tuple[tuple[str, str], ...] = ()
    training_runtime_provenance_verified: bool = False
    scientific_acceptance_permitted: bool = False


def audit_archived_git_sources(
    repo_root: Path, manifest: dict[str, Any],
) -> HistoricalSourceAudit:
    if not isinstance(repo_root, Path) or not isinstance(manifest, dict):
        raise ValueError("repo path and archived JSON object required")
    root = repo_root.resolve(strict=True)
    git_commit = manifest.get("git_commit")
    if not isinstance(git_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", git_commit):
        raise ValueError("immutable full-length Git commit required")
    if manifest.get("heldout_rollouts") != 0 or manifest.get("scientific_evidence") is not False:
        raise ValueError("historical audit only permits sealed exploratory evidence")
    entries = manifest.get("source_fingerprints")
    if not isinstance(entries, list) or len(entries) != len(ARCHIVED_SOURCE_PATHS):
        raise ValueError("historical source file inventory is incomplete")
    names = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid fingerprint item")
        name, declared = entry.get("path"), entry.get("sha256")
        if not isinstance(name, str) or not isinstance(declared, str):
            raise ValueError("source fingerprint fields are required")
        if not re.fullmatch(r"[0-9a-f]{64}", declared):
            raise ValueError("invalid SHA256 in historical source manifest")
        names.append(name)
    if len(set(names)) != len(names) or set(names) != set(ARCHIVED_SOURCE_PATHS):
        raise ValueError("unregistered, duplicate or missing historical source paths")
    ordered = {entry["path"]: entry["sha256"] for entry in entries}
    failures: list[str] = []
    changed: list[str] = []
    eol_modes: list[tuple[str, str]] = []
    for name in ARCHIVED_SOURCE_PATHS:
        expected = ordered[name]
        result = subprocess.run(
            ["git", "-C", str(root), "show", f"{git_commit}:{name}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        blob = result.stdout
        if not result.returncode and sha256(blob).hexdigest() == expected:
            eol_modes.append((name, "raw-git-blob"))
        elif (not result.returncode and b"\r" not in blob
              and sha256(blob.replace(b"\n", b"\r\n")).hexdigest() == expected):
            # Git text checkout may materialize LF blobs as CRLF bytes on
            # Windows. Log the *explicit* transform; never silently pretend
            # historical file bytes equal the raw Git object.
            eol_modes.append((name, "git-lf-to-windows-crlf"))
        else:
            failures.append(f"historical commit source SHA256 mismatch: {name}")
        file = (root / name).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            changed.append(name)
        elif sha256(file.read_bytes()).hexdigest() != expected:
            changed.append(name)
    return HistoricalSourceAudit(
        git_commit, not failures, len(ARCHIVED_SOURCE_PATHS),
        tuple(changed), tuple(failures), tuple(eol_modes),
    )


def audit_historical_manifest_file(
    repo_root: Path, manifest_path: Path,
) -> HistoricalSourceAudit:
    if not isinstance(manifest_path, Path):
        raise ValueError("manifest path required")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    return audit_archived_git_sources(repo_root, document)
