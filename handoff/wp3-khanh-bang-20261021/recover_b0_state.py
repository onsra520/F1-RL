#!/usr/bin/env python3
"""Conservative B0 campaign-state reconciliation on the pinned WP3 source.

The original source commit and complete TRAIN telemetry are mandatory. This
helper never launches training. It refuses to change state unless B0's full
run evidence independently validates against the original immutable manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys


PINNED_SOURCE = "fe83f517479f3b990028d6409304fae83b8199a9"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Checkout at exact frozen WP3 source commit")
    parser.add_argument("--evidence", type=Path, required=True, help="Extracted attempt-v8 full evidence directory")
    parser.add_argument("--apply", action="store_true", help="Explicitly reconcile on-disk campaign state (no training)")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    evidence = args.evidence.resolve(strict=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if head != PINNED_SOURCE:
        raise SystemExit(f"Refusing mismatched source HEAD {head}; expected {PINNED_SOURCE}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip():
        raise SystemExit("Refusing dirty pinned training source checkout")
    sys.path.insert(0, str(source))
    from f1rl.local_demo.campaign import (
        _CampaignLock, _append_campaign_event, _atomic_json, _load_campaign, _validate_or_reuse_run,
    )

    with _CampaignLock(evidence / ".campaign.lock"):
        manifest, config, schedule, state, _ = _load_campaign(source, evidence)
        if state["status"] != "RUNNING" or state.get("current_model_id") != "B-seed0":
            raise SystemExit(f"Unexpected state: {state['status']}, current={state.get('current_model_id')}")
        if state["completed_model_ids"] != ["C-seed0", "A-seed0"]:
            raise SystemExit(f"Unexpected completed sequence: {state['completed_model_ids']}")
        if state["models"]["B-seed0"]["status"] != "running":
            raise SystemExit("B0 state is not the known interrupted commit boundary")
        if "training-telemetry.jsonl" not in {p.name for p in (evidence / "runs" / "B-seed0").iterdir()}:
            raise SystemExit("Full original TRAIN telemetry is missing; checkpoint files are insufficient")
        # All three model runs are revalidated against immutable source/track/seed
        # identities before reconciling the interrupted state.
        valid = {}
        for model_id in ("C-seed0", "A-seed0", "B-seed0"):
            expected = next(row for row in schedule if row["model_id"] == model_id)
            run = evidence / expected["run_relative_path"]
            validation = _validate_or_reuse_run(source, run, expected, manifest, config)
            if validation is None or validation["status"] != "ENGINEERING_PASS":
                raise SystemExit(f"{model_id}: full validation failed")
            valid[model_id] = validation
            print(f"{model_id}: full evidence ENGINEERING_PASS", flush=True)

        events = [json.loads(line) for line in (evidence / "campaign-events.jsonl").read_text(encoding="utf-8").splitlines()]
        b0_pass_events = [event for event in events if event.get("event") == "model_engineering_pass" and event.get("model_id") == "B-seed0"]
        if b0_pass_events:
            raise SystemExit("B0 pass event already exists; manual audit required")
        if not args.apply:
            print("DRY RUN: No files changed. Re-run with --apply to reconcile state, then run WP3 resume.")
            return 0

        backup = evidence / ("campaign-state.before-handoff-recovery-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json")
        if backup.exists():
            raise SystemExit(f"Backup already exists: {backup}")
        shutil.copy2(evidence / "campaign-state.json", backup)
        state["models"]["B-seed0"].update(status="ENGINEERING_PASS", validation=valid["B-seed0"],
                                         output_path="runs/B-seed0", recovery_note="validated_completed_run_after_orchestrator_exit")
        state["completed_model_ids"].append("B-seed0")
        state["completed_training_timesteps"] = len(state["completed_model_ids"]) * config["training"]["timesteps_per_model"]
        state["current_model_id"] = None
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(evidence / "campaign-state.json", state)
        _append_campaign_event(evidence / "campaign-events.jsonl", {
            "event": "model_engineering_pass", "model_id": "B-seed0",
            "learning_timesteps": valid["B-seed0"]["learning_timesteps"],
            "optimizer_steps": valid["B-seed0"]["optimizer_steps"],
            "telemetry_sha256": valid["B-seed0"]["telemetry"]["sha256"],
            "recovered_from_verified_existing_run": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        print(f"RECOVERED: 3/12 official completed. Backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
