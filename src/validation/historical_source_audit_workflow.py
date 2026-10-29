"""Read-only source provenance report for *historical* WP7 PPO archives.

Usage:
    python scripts/wp7_historical_source_audit.py --archive-run RUN --output OUT.json

Never rewrites a training archive, opens HELD-OUT, runs PPO or asserts
Scientific Acceptance. An exact Git commit source match cannot recover
unknown training-time Python packages or GPU settings.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from src.validation.wp7_historical_provenance import audit_historical_manifest_file

def main(repository_root: Path, argv: list[str] | None = None) -> int:
    root = Path(repository_root).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    target = args.output.resolve()
    if target.is_relative_to(root):
        parser.error("audit output must be outside the source worktree")
    if target.exists():
        parser.error("refuse to overwrite existing audit report")
    reports = []
    for run in args.archive_run:
        manifest = run.resolve(strict=True) / "evaluations" / "manifest.json"
        result = audit_historical_manifest_file(root, manifest)
        reports.append({"run_id": run.name, **asdict(result)})
    payload = {
        "schema": "wp7-historical-source-audit-v1",
        "scientific_acceptance_permitted": False,
        "training_permitted": False,
        "heldout_permitted": False,
        "archive_runs": reports,
    }
    document = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(document)
    print(json.dumps({
        "output": str(target),
        "sha256": sha256(document).hexdigest(),
        "archived_sources_verified": all(a["archived_source_verified"] for a in reports),
        "run_status": [
            {
                "run_id": r["run_id"],
                "source_commit": r["archived_git_commit"],
                "historical_source_verified": r["archived_source_verified"],
                "worktree_drift": r["current_worktree_drifts"],
                "failures": r["failures"],
                "training_runtime_provenance_verified": False,
            } for r in reports
        ],
        "scientific_acceptance_permitted": False,
    }, indent=2))
    return 0 if all(a["archived_source_verified"] for a in reports) else 2
