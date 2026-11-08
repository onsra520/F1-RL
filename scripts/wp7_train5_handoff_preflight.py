"""Read-only, portable TRAIN-5 handoff source lock and no-training preflight.

No geometry parsing, PPO loading, runtime replay, seed allocation, or admission.
The four independent Step-3 witness files are verified only if an explicit
external evidence root is provided; otherwise they are reported NOT_VERIFIED.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

REG = "configs/research/wp7-train5-paired-v2-registration.json"
BASELINE = "2417570f32f6879c2651bce37d88fd6ac83a2ea4"
REGISTERED_BASELINE = "1de279c37e62a27531d661480d20fa6e095c4e84"
TRAIN5 = (
    "straight-sprint-v1",
    "slalom-v1",
    "right-turn-train-v2",
    "ramp-v1",
    "left-turn-train-v2",
)
HELDOUT = ("left-turn-v1", "offset-slalom-v1")
PROPOSED_GROUPS = {"train5_from_scratch", "train5_from_frozen_checkpoint"}


class HandoffPreflightError(ValueError):
    pass


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise HandoffPreflightError(message)


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _file_under(root: Path, relative: str) -> Path:
    _require(
        type(relative) is str
        and relative
        and not relative.startswith(("/", "\\"))
        and "\\" not in relative
        and not Path(relative).is_absolute()
        and ".." not in Path(relative).parts,
        f"non-portable or unsafe source path: {relative}",
    )
    path = root / relative
    _require(
        path.is_file() and not path.is_symlink()
        and path.resolve().is_relative_to(root.resolve()),
        f"missing or escaping source path: {relative}",
    )
    return path


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True, text=True, check=False,
    )


def audit(root: Path, evidence_root: Path | None = None) -> dict:
    root = root.resolve()
    regfile = _file_under(root, REG)
    reg = json.loads(regfile.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    _require(type(reg) is dict, "registration is not an object")
    _require(reg.get("status") == "draft_not_authorized", "registration not in draft state")
    for name in ("training_permission", "smoke_permission", "heldout_permission"):
        _require(reg.get(name) is False, f"{name} must remain false")
    for name in ("global_seed_registry_receipt", "owner_approval"):
        _require(reg.get(name) is None, f"{name} must remain null")
    _require(reg.get("seed_audit_status") == "PROPOSED_UNVERIFIED_GLOBAL",
             "global seed provenance must remain proposed/unverified")
    _require(reg.get("train_ids") == list(TRAIN5), "TRAIN-5 roster drift")
    _require(reg.get("heldout_ids") == list(HELDOUT), "HELD-OUT metadata drift")
    _require(reg.get("baseline_head") == REGISTERED_BASELINE, "registration source anchor drift")
    _require(reg.get("runtime_profile") == {
        "physics_version": "spatial_road_edge_partial_lifecycle_v4",
        "race_version": "wp7_directed_lifecycle_crash_contract_v4",
        "terminal_contract": "wp3_physical_crash_terminal_v1",
        "max_abs_tilt_rad": 0.7,
    }, "v4 runtime profile drift")
    _require(reg.get("proposed_design", {}).get("proposed_development_cells") == 60
             and reg.get("proposed_design", {}).get("proposed_development_episodes") == 3600,
             "development planning counts drift")
    groups = reg.get("proposed_training_seeds")
    _require(type(groups) is dict and set(groups) == PROPOSED_GROUPS, "seed group drift")
    seedvalues = []
    for name, rows in groups.items():
        _require(type(rows) is list and len(rows) == 3, f"{name} must have 3 independent bundles")
        for bundle in rows:
            _require(type(bundle) is dict
                     and set(bundle) == {"ppo", "sampling", "perturbation"},
                     "seed bundle schema drift")
            seedvalues.extend(bundle.values())
    _require(len(seedvalues) == 18
             and len(set(seedvalues)) == 18
             and all(type(x) is int and 0 <= x < 2**32 for x in seedvalues),
             "seed proposal must contain exactly 18 distinct uint32 values")
    doc = {k: v for k, v in reg.items() if k != "preflight_digest"}
    actual_digest = sha256(json.dumps(
        doc, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    _require(reg.get("preflight_digest") == actual_digest, "registration canonical digest drift")

    sources = reg.get("source_sha256")
    witnesses = reg.get("evidence_sha256")
    _require(type(sources) is dict and len(sources) == 44,
             "expected exactly 44 registered source inputs")
    _require(type(witnesses) is dict and len(witnesses) == 4,
             "expected exactly 4 registered external witness inputs")
    bridge_path = _file_under(
        root, "configs/research/wp7-train5-paired-v2-git-blob-lock.json"
    )
    bridge = json.loads(bridge_path.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    _require(bridge.get("schema") == "f1rl_train5_source_eol_gitblob_bridge_v1",
             "portable Git blob bridge schema mismatch")
    _require(bridge.get("source_commit") == "74690534949c657bde9fa18b7cf4c895862739b7",
             "portable Git blob bridge source anchor drift")
    registration_git = subprocess.run(
        ["git", "-C", str(root), "show", "HEAD:" + REG],
        capture_output=True, check=False,
    )
    _require(registration_git.returncode == 0, "tracked registration missing")
    _require(
        sha256(registration_git.stdout).hexdigest()
        == bridge.get("registration_git_blob_sha256"),
        "portable Git blob bridge registration canonical SHA drift",
    )
    registration_bytes = regfile.read_bytes()
    _require(
        sha256(registration_bytes).hexdigest()
        in (bridge.get("registration_sha256"),
            bridge.get("registration_git_blob_sha256"))
        and (registration_bytes == registration_git.stdout
             or registration_bytes.replace(b"\r\n", b"\n") == registration_git.stdout),
        "portable Git blob bridge registration SHA drift",
    )
    entries = bridge.get("source_blobs")
    _require(type(entries) is dict and set(entries) == set(sources),
             "portable Git blob bridge file roster drift")
    direct_sha_count = 0
    eol_bridge_count = 0
    for relative, expected in sources.items():
        _require("evaluation/" not in relative and "heldout" not in relative.lower(),
                 f"refuse to read HELD-OUT geometry: {relative}")
        _require(type(expected) is str and len(expected) == 64,
                 f"invalid expected source digest: {relative}")
        record = entries[relative]
        _require(record.get("registered_windows_worktree_sha256") == expected,
                 f"registered Windows receipt mismatch: {relative}")
        path = _file_under(root, relative)
        contents = path.read_bytes()
        actual = sha256(contents).hexdigest()
        tracked = subprocess.run(
            ["git", "-C", str(root), "show", "HEAD:" + relative],
            capture_output=True, check=False,
        )
        _require(tracked.returncode == 0, f"tracked source missing: {relative}")
        git_blob = tracked.stdout
        _require(sha256(git_blob).hexdigest() == record.get("git_blob_sha256"),
                 f"pinned Git blob SHA mismatch: {relative}")
        if actual == expected:
            direct_sha_count += 1
            _require(contents == git_blob or contents.replace(b"\r\n", b"\n") == git_blob,
                     f"registered byte source not Git-equivalent: {relative}")
        else:
            _require(record.get("relation") in ("byte_identical", "CRLF_to_LF_only"),
                     f"unexplained source byte mismatch: {relative}")
            _require(contents == git_blob or (
                b"\0" not in git_blob and b"\r" not in git_blob
                and contents == git_blob.replace(b"\n", b"\r\n")
            ),
                     f"source content differs from pinned Git blob: {relative}")
            eol_bridge_count += 1

    baseline = _git(root, "merge-base", "--is-ancestor", BASELINE, "HEAD")
    _require(baseline.returncode == 0, "historical source baseline not an ancestor")
    head = _git(root, "rev-parse", "HEAD")
    _require(head.returncode == 0, "unable to identify Git HEAD")
    witness_status = "NOT_VERIFIED_EXTERNAL_ROOT_NOT_PROVIDED"
    if evidence_root is not None:
        _require(evidence_root.resolve() != root and not root in evidence_root.resolve().parents,
                 "evidence root must remain outside checkout")
        for relative, expected in witnesses.items():
            _require(type(expected) is str and len(expected) == 64,
                     "invalid external witness hash")
            _require(_sha(_file_under(evidence_root.resolve(), relative)) == expected,
                     f"external Step-3 witness mismatch: {relative}")
        witness_status = "FOUR_EXTERNAL_WITNESSES_SHA256_VERIFIED"

    return {
        "result": "PASS_SOURCE_LOCK_NO_TRAINING",
        "source_commit": head.stdout.strip(),
        "historical_baseline_ancestor": BASELINE,
        "source_file_git_blob_hashes_verified": len(sources),
        "registered_windows_sha256_direct_matches": direct_sha_count,
        "registered_windows_sha256_via_pinned_eol_bridge": eol_bridge_count,
        "external_witness_status": witness_status,
        "external_witness_count": len(witnesses) if evidence_root else 0,
        "proposed_seed_count": len(seedvalues),
        "seed_global_status": "PROPOSED_UNVERIFIED_GLOBAL",
        "training_permission": False,
        "smoke_permission": False,
        "heldout_permission": False,
        "scientific_release_permission": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--evidence-root", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.root, args.evidence_root)
    except (HandoffPreflightError, OSError, ValueError) as exc:
        print(json.dumps({"result": "FAIL_CLOSED", "error": str(exc)}))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
