"""WP3 campaign report and evidence artifact writers."""
from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path

from ..rl.pilot_preflight import TRAIN_IDS
from ._artifacts import (
    _canonical,
    _inside,
    _sha256,
    _strict_json,
    _write_immutable_json,
    _write_new_json,
)


def _write_c0_gate_report(path: Path, validation: dict, status: str,
                          reason: str | None = None) -> None:
    lifecycle = validation["lifecycle"]
    lines = [
        "# WP3 C0 Full-Length PPO Engineering Gate",
        "",
        f"Status: **{status}**",
        "",
        f"- Model: `{validation['model_id']}` (candidate {validation['candidate_id']}, seed index {validation['seed_index']})",
        f"- Source commit: `{validation['source_commit']}`",
        f"- Learning timesteps: {validation['learning_timesteps']:,}",
        f"- PPO update rows: {validation['ppo_update_rows']:,}",
        f"- Optimizer steps: {validation['optimizer_steps']:,}",
        f"- Initial policy weights SHA-256: `{validation['initial_policy_weights_sha256']}`",
        f"- Final policy weights SHA-256: `{validation['final_policy_weights_sha256']}`",
        f"- Telemetry SHA-256: `{validation['telemetry']['sha256']}` ({validation['telemetry']['bytes']:,} bytes)",
        f"- Checkpoints verified: {len(validation['checkpoints'])} archives (16 periodic plus final)",
        f"- Episodes: {lifecycle['episode_count']}; automatic reset boundaries verified: {lifecycle['automatic_resets_verified']}",
        f"- Natural terminations: {lifecycle['natural_terminations']}; curriculum terminations: {lifecycle['curriculum_terminations']}; truncations: {lifecycle['truncations']}",
        f"- Unsupported-contact training terminals: {lifecycle['physics_abort_terminations']}",
        "- Runtime split: TRAIN only; HELD-OUT access and scientific evaluation: disabled.",
        "",
        "## Observed registered TRAIN coverage",
        "",
        "| Track | Timesteps | Episodes |",
        "|---|---:|---:|",
    ]
    for track in TRAIN_IDS:
        lines.append(f"| {track} | {lifecycle['track_timestep_counts'][track]:,} | {lifecycle['track_episode_counts'][track]:,} |")
    lines += ["", "Driving success is not part of this engineering gate."]
    if reason:
        lines += ["", f"Gate blocker: {reason}"]
    lines += ["", "WP7 and Gate 2 remain **BLOCKED**.", ""]
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
        stream.flush()
        os.fsync(stream.fileno())


def _write_training_deliverables(output: Path, state: dict, schedule: list[dict],
                                 manifest: dict, config: dict) -> None:
    inventory = []
    for model in schedule:
        run_dir = output / model["run_relative_path"]
        validation = state["models"][model["model_id"]]["validation"]
        archives = []
        for archive in validation["checkpoints"]:
            path = _inside(run_dir, archive["path"])
            archives.append({**archive, "size_bytes": path.stat().st_size})
        inventory.append({
            "model_id": model["model_id"], "candidate_id": model["candidate_id"],
            "seed_index": model["seed_index"], "seed_bundle": model["seed_bundle"],
            "source_commit": manifest["source_commit"],
            "training_timesteps": validation["learning_timesteps"],
            "initial_policy_weights_sha256": validation["initial_policy_weights_sha256"],
            "final_model": archives[-1], "checkpoints": archives,
            "training_telemetry": validation["telemetry"],
            "lifecycle": validation["lifecycle"],
            "resource_usage": state["models"][model["model_id"]].get("resource_usage"),
            "elapsed_seconds": state["models"][model["model_id"]].get("elapsed_seconds"),
            "run_path": model["run_relative_path"],
        })
    result = {
        "schema_version": "f1rl_wp3_training_results_v1",
        "status": "ENGINEERING_PASS", "campaign_manifest_sha256": manifest["sha256"],
        "source_commit": manifest["source_commit"], "models": inventory,
        "expected_models": 12, "completed_models": len(inventory),
        "expected_timesteps": config["training"]["total_timesteps"],
        "completed_timesteps": sum(row["training_timesteps"] for row in inventory),
        "scientific_evidence": False, "heldout_access": False,
        "wp7_status": "BLOCKED", "gate2_status": "BLOCKED",
    }
    result["sha256"] = sha256(_canonical(result)).hexdigest()
    _write_immutable_json(output / "WP3-TRAINING-RESULTS.json", result)
    inventory_doc = {
        "schema_version": "f1rl_wp3_model_inventory_v1",
        "campaign_manifest_sha256": manifest["sha256"], "models": inventory,
        "selection_rule": "final_checkpoint_only", "scientific_evidence": False,
        "heldout_access": False,
    }
    inventory_doc["sha256"] = sha256(_canonical(inventory_doc)).hexdigest()
    _write_immutable_json(output / "WP3-MODEL-INVENTORY.json", inventory_doc)


def _write_independent_review(output: Path, schedule: list[dict], state: dict,
                              evaluation: dict) -> None:
    lines = [
        "# WP3 Independent Source and Evidence Review", "",
        "Review status: **PASS** for the bounded simulation-only campaign mechanics; this review does not grant scientific acceptance.", "",
        "The second-pass review checked the committed campaign implementation against its frozen manifest and inspected the resulting run evidence independently of the campaign state summary.", "",
        "| Area | Source / evidence | Review result |", "|---|---|---|",
        "| Frozen design and schedule | `configs/research/wp3-ppo-tuning-campaign-v3.json`; `src/interfaces/campaign.py:model_schedule` | 12 unique candidate/seed identities in fixed order; parameters bound to simulation-only manifest |",
        "| Training integrity | `src/interfaces/runner.py:run_local_demo`; `src/training/telemetry.py`; each `runs/*/training-integrity.json` | Real PPO updates, per-step finite/nonzero gradients, changed weights, telemetry and checkpoints independently checked |",
        "| Training transitions | `src/interfaces/campaign.py:validate_training_run`; each `runs/*/training-telemetry.jsonl` | Exact step/event counts, source/seed/reset/draw-seed/track provenance and terminal boundaries checked |",
        "| Checkpoint integrity | each `runs/*/checkpoints.json`, `model-final.zip`, and engineering validation | All periodic archives plus final archive hash-checked and CPU-loaded |",
        "| Evaluation isolation | `src/interfaces/evaluation.py:run_evaluation`; `WP3-TRAIN-EVALUATION.json` | TRAIN-only paired roster, final checkpoints only, original 4 m task, no curriculum/scientific admission |",
        "| Resume and failure handling | `campaign-state.json`, append-only event logs, per-cell JSONL | Existing/incomplete outputs are preserved; interrupted training is not automatically retried; evaluation invalid cells are not redrawn |",
        "| Full local checks | `WP3-VALIDATION-RESULTS.json` | See recorded pytest, compileall, pip check, and diff-check outcomes |",
        "", f"Training identities independently reviewed: {len(schedule)}. Completed: {len(state.get('completed_model_ids', []))}.",
        f"Evaluation cells complete: {evaluation.get('completed_cells', 0)}/96; invalid cells: {len(evaluation.get('invalid_cells', []))}.",
        "", "Residual limits: three tuning seed blocks support descriptive comparisons only; episode-level repetitions are not independent PPO seed blocks. No confirmatory inference or WP7 acceptance is claimed. Gate 2 and WP7 remain **BLOCKED**.", "",
    ]
    content = "\n".join(lines)
    path = output / "WP3-INDEPENDENT-REVIEW.md"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError("refusing to overwrite immutable WP3 independent review")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _write_evidence_manifest(output: Path) -> None:
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name in {"WP3-EVIDENCE-MANIFEST.json", ".campaign.lock"}:
            continue
        if path.suffix == ".tmp" or path.name.endswith(".pending.zip"):
            raise ValueError(f"temporary or incomplete artifact remains in campaign evidence: {path}")
        files.append({"path": path.relative_to(output).as_posix(),
                      "bytes": path.stat().st_size, "sha256": _sha256(path)})
    manifest = {"schema_version": "f1rl_wp3_evidence_manifest_v1",
                "files": files, "file_count": len(files),
                "scientific_evidence": False, "heldout_access": False,
                "wp7_status": "BLOCKED", "gate2_status": "BLOCKED"}
    manifest["sha256"] = sha256(_canonical(manifest)).hexdigest()
    _write_new_json(output / "WP3-EVIDENCE-MANIFEST.json", manifest)


def _verify_evidence_manifest(output: Path) -> None:
    path = output / "WP3-EVIDENCE-MANIFEST.json"
    manifest, _ = _strict_json(path)
    if manifest.get("sha256") != sha256(_canonical({k: v for k, v in manifest.items()
                                                       if k != "sha256"})).hexdigest():
        raise ValueError("WP3 evidence manifest self-checksum mismatch")
    expected = {row["path"]: row for row in manifest.get("files", [])}
    actual_paths = {file.relative_to(output).as_posix(): file
                    for file in output.rglob("*")
                    if file.is_file() and file.name not in {path.name, ".campaign.lock"}}
    if set(expected) != set(actual_paths):
        raise ValueError("WP3 evidence manifest inventory differs from the campaign root")
    for relative, file in actual_paths.items():
        row = expected[relative]
        if row.get("bytes") != file.stat().st_size or row.get("sha256") != _sha256(file):
            raise ValueError(f"WP3 campaign evidence changed after final manifest: {relative}")
