"""WP3 Issue #51: independent source-bound Eval-only coordinator.

Usage:
    python -m scripts.evaluation.wp3_eval_only_coordinator_v4 preflight
    python -m scripts.evaluation.wp3_eval_only_coordinator_v4 smoke --runtime-profile wp3_full06_physics_v4 --out D:/unique-new-run-dir
    python -m scripts.evaluation.wp3_eval_only_coordinator_v4 full  # intentionally fail-closed
"""
from __future__ import annotations
from pathlib import Path
import argparse, json, sys

ROOT = Path(__file__).resolve().parents[3]
from f1rl.research.wp3_eval_only_coordinator import (
    FULL06_RUNTIME_PROFILE, RUNTIME_PROFILE_SPECS,
    preflight, forbid_registered_full_eval, run_engineering, runtime_profile_spec)

HISTORY = Path(r"D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1")

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "smoke", "full"))
    parser.add_argument("--runtime-profile", choices=tuple(RUNTIME_PROFILE_SPECS),
                        default=FULL06_RUNTIME_PROFILE)
    parser.add_argument("--history", type=Path, default=HISTORY)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run-id", default="WP3-ENG-ISSUE51-20261024-REPLAY2")
    parser.add_argument("--episode", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        check, roster, provenance = preflight(ROOT, args.history)
        requested_runtime = runtime_profile_spec(args.runtime_profile)
        print(json.dumps({
            "status": "INPUT_PROVENANCE_PASS_FULL_EVAL_STILL_BLOCKED",
            "evaluation_source": check["current_evaluation_source_commit"],
            "requested_runtime_profile": args.runtime_profile,
            "expected_effective_runtime": {
                key: value for key, value in requested_runtime.items() if key != "profile"},
            "frozen_models": len(provenance["models"]),
            "original_roster_cells": len(roster["seed_rows"]),
            "original_roster_expected_episodes": roster["expected_episodes"],
            "heldout_access": False,
        }, sort_keys=True))
        return 0
    if args.command == "full":
        try:
            forbid_registered_full_eval(ROOT, args.history)
        except RuntimeError as error:
            print(f"FULL_EVAL_BLOCKED: {error}", file=sys.stderr)
            return 3
        return 4
    if args.out is None:
        parser.error("--out is required for a fresh engineering smoke")
    report = run_engineering(ROOT, args.history, args.out, args.run_id,
                             episode_index=args.episode,
                             runtime_profile=args.runtime_profile)
    return 0 if report["invalid"] == 0 else 2

if __name__ == "__main__":
    raise SystemExit(main())
