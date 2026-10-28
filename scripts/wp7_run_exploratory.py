"""Explicitly authorized TRAIN-only exploratory PPO calibration/confirmation.

No frozen-science claims. With --smoke, trains a separate 2048-step disposable
diagnostic without final/milestone evidence; no HELD-OUT rollouts are possible.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import sys

from src.learning.pilot_milestones import PilotMilestoneRecorder, verify_pilot_evidence_manifest
from src.learning.pilot_preflight import TRAIN_IDS, build_pilot_preflight
from src.learning.reconstruction import RecoveryCandidate, build_training_scenarios
from src.learning.training.config import TrainingRunConfig
from src.learning.training.runner import TrainingRunner


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("calibration", "confirmation"))
    parser.add_argument("--accept-review-risks", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--artifacts-root", type=Path, required=True)
    args = parser.parse_args()

    if not args.accept_review_risks:
        parser.error("exploratory training requires explicit --accept-review-risks")
    preflight = build_pilot_preflight(
        repository_root=root,
        candidate_path=root/"configs/recovery/wp7-c2-candidate.yml",
        runtime_config_path=root/"configs/recovery/wp7-pilot-200k.yml",
    )
    if preflight["training_track_ids"] != list(TRAIN_IDS):
        raise RuntimeError("TRAIN track boundary changed")
    print(json.dumps({
        "phase":"preflight",
        "scientific_evidence":False,
        "role":args.role,
        "smoke":args.smoke,
        "review_blockers":preflight["review_blockers"],
    }),flush=True)
    config = TrainingRunConfig.from_yaml(root/"configs/recovery/wp7-pilot-200k.yml")
    seed = 0 if args.role == "calibration" else 1
    location = args.artifacts_root.resolve()
    if location == root or location.is_relative_to(root):
        raise ValueError("artifacts_root must be outside the Git worktree")
    config = replace(
        config, ppo=replace(config.ppo,seed=seed),
        sampling_seed=seed, artifacts_root=location,
    )
    scenarios=build_training_scenarios(
        RecoveryCandidate.from_yaml(root/"configs/recovery/wp7-c2-candidate.yml")
    )
    if tuple(s.track_id for s in scenarios.scenarios)!=TRAIN_IDS:
        raise RuntimeError("TRAIN-only scenario boundary failed")
    if args.smoke:
        config=replace(config,total_timesteps=2_048,checkpoint_every_steps=None)
        runner=TrainingRunner(config)
    else:
        recorder=PilotMilestoneRecorder(
            repository_root=root,
            candidate_path=root/"configs/recovery/wp7-c2-candidate.yml",
            runtime_config_path=root/"configs/recovery/wp7-pilot-200k.yml",
            episodes_per_track=1,
            exploratory_mode=True,
        )
        def record_checkpoint(archive: Path, steps: int) -> None:
            recorder.on_checkpoint(archive, steps)
            print(json.dumps({
                "phase":"milestone_verified","actual_steps":steps,
                "milestone":recorder.milestones_recorded[-1],
                "model":str(archive),"scientific_evidence":False,
            }), flush=True)

        def record_final(archive: Path, steps: int) -> None:
            recorder.on_final(archive, steps)
            print(json.dumps({
                "phase":"final_milestone_verified","actual_steps":steps,
                "milestone":recorder.milestones_recorded[-1],
                "model":str(archive),"scientific_evidence":False,
            }), flush=True)

        runner=TrainingRunner(
            config,checkpoint_observer=record_checkpoint,
            final_model_observer=record_final,
        )
    print(json.dumps({
        "phase":"training_start",
        "requested_steps":config.total_timesteps,
        "ppo_seed":seed,"sampling_seed":seed,
        "artifacts_root":str(location),
        "python":platform.python_version(),
        "scientific_evidence":False,
    }),flush=True)
    result=runner.run_multi(scenarios)
    if not args.smoke:
        verified=verify_pilot_evidence_manifest(result.run_directory)
        print(json.dumps({"phase":"postrun_checksum_verification",**verified}),flush=True)
    print(json.dumps({
        "phase":"training_complete",
        "run_directory":str(result.run_directory),
        "checkpoint_paths":[str(x) for x in result.checkpoint_paths],
        "final_model":str(result.final_model_path),
        "requested_steps":result.requested_total_timesteps,
        "actual_steps":result.actual_total_timesteps,
        "scientific_evidence":False,
    }),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
