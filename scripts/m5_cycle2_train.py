"""Freeze Cycle 2 plans and run its single bounded PPO fine-tuning attempt."""
from __future__ import annotations

from hashlib import sha256
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from f1rl.local_demo.admission import _source_identity
from f1rl.local_demo.runner import run_m5_cycle2
from m5_c0_diagnostic import canonical_sha256, file_sha256
from m5_cycle2_evaluate import freeze_plan as freeze_evaluation_plan, verify_plan

PROFILE = ROOT / "configs/research/m5-cycle2-c0-slalom-v1.json"
STD050 = Path(r"D:\Projects\.archives\F1-RL\demo-completion\m5-improvement-cycle1-20261111\C-seed0-logstd-scale-0.5.zip")
CAMPAIGN_PLAN = Path(r"D:\Projects\.archives\F1-RL\demo-completion\m5-multicandidate-screening-20261111-reprocheck\CAMPAIGN-PLAN.json")
EXPECTED_PARENT_SHA = "b6de69382523b9dc5309a6cffb230afd31f9b0206d3381dff723abc69ebffa86"
TRACKS = ["straight-sprint-v1", "slalom-v1", "right-turn-train-v2", "ramp-v1",
          "left-turn-train-v2"]


def make_training_plan(root: Path, evidence_root: Path, evaluation_plan: Path) -> dict:
    if evidence_root.exists():
        raise FileExistsError(evidence_root)
    if file_sha256(STD050) != EXPECTED_PARENT_SHA:
        raise ValueError("locked C0 std050 checkpoint SHA mismatch")
    profile_bytes = PROFILE.read_bytes()
    profile = json.loads(profile_bytes)
    evaluation = verify_plan(evaluation_plan)
    commit, tree_sha = _source_identity(root)
    if evaluation["source_commit"] != commit:
        raise ValueError("training and checkpoint-selection plans must share the exact source commit")
    plan = {
        "schema_version": "f1rl_m5_cycle2_training_plan_v1",
        "source_commit": commit, "source_tree_sha256": tree_sha,
        "source_branch": subprocess.run(["git", "branch", "--show-current"], cwd=root,
                                         check=True, capture_output=True, text=True).stdout.strip(),
        "source_file_sha256": {
            name: file_sha256(root / name) for name in (
                "f1rl/local_demo/runner.py", "f1rl/local_demo/admission.py",
                "f1rl/local_demo/m5_cycle2_telemetry.py", "f1rl/learning/algorithms/ppo.py",
                "scripts/m5_multicandidate_screening.py", "scripts/m5_cycle2_evaluate.py",
                "configs/research/m5-cycle2-c0-slalom-v1.json")},
        "profile_sha256": sha256(profile_bytes).hexdigest(),
        "parent_checkpoint_path": str(STD050.resolve()),
        "parent_checkpoint_sha256": EXPECTED_PARENT_SHA,
        "training_authorized_by_owner_request_sha256": profile["owner_request_sha256"],
        "track_ids": TRACKS,
        "sampling_policy": profile["sampling_policy"],
        "track_asset_sha256": {track: file_sha256(
            root / "assets/tracks/training" / f"{track}.json") for track in TRACKS},
        "sampling_seed": profile["sampling_seed"],
        "max_additional_timesteps": profile["max_additional_timesteps"],
        "checkpoint_interval": profile["checkpoint_interval"],
        "rollout_length": profile["rollout_length"], "n_envs": profile["n_envs"],
        "device": profile["device"], "max_wall_seconds": profile["max_wall_seconds"],
        "evaluation_plan_path": str(evaluation_plan.resolve()),
        "evaluation_plan_sha256": evaluation["sha256"],
        "selection_gates": {
            "deterministic_slalom_minimum": "8/10",
            "stochastic_slalom_combined_minimum": "12/15",
            "stochastic_slalom_fixed_minimum": "8/10",
            "stochastic_slalom_fresh_development_minimum": "4/5",
            "other_stochastic_tracks_minimum": "15/15_each",
            "population_inference": False,
        },
        "heldout_access": False, "scientific_evidence": False,
    }
    plan["sha256"] = canonical_sha256(plan)
    return plan


def evaluate_after_checkpoint(checkpoint: Path, metadata: dict, plan: dict,
                              evidence_root: Path) -> bool:
    # The training process pauses while this isolated read-only evaluation runs.
    plan_path = Path(plan["evaluation_plan_path"])
    evaluation_plan = verify_plan(plan_path)
    if evaluation_plan["sha256"] != plan["evaluation_plan_sha256"]:
        raise ValueError("paired evaluation roster differs from the frozen training plan")
    checkpoint_dir = evidence_root / "checkpoint-evaluations" / f"step-{metadata['timestep']}"
    command = [sys.executable, str(ROOT / "scripts/m5_cycle2_evaluate.py"), "--evaluate",
               str(plan_path), str(checkpoint), str(checkpoint_dir)]
    try:
        child = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                               timeout=8 * 60 * 60, check=False)
    except subprocess.TimeoutExpired as error:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "worker-timeout.txt").write_text(
            f"timeout_seconds={error.timeout}\ncheckpoint_sha256={metadata['checkpoint_sha256']}\n",
            encoding="utf-8")
        raise RuntimeError("paired checkpoint evaluation timed out; fine-tuning stopped") from error
    if child.returncode != 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "worker-failure.txt").write_text(
            f"returncode={child.returncode}\nstdout={child.stdout[-3000:]}\nstderr={child.stderr[-5000:]}\n",
            encoding="utf-8")
        raise RuntimeError("paired checkpoint evaluation failed; fine-tuning stopped")
    report_path = checkpoint_dir / "EVALUATION-REPORT.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report["checkpoint_sha256"] != metadata["checkpoint_sha256"]
            or report["complete_cases"] != 85):
        raise ValueError("checkpoint evaluation did not complete the exact paired roster")
    deterministic_dir = evidence_root / "deterministic-slalom" / f"step-{metadata['timestep']}"
    det_command = [sys.executable, str(ROOT / "scripts/m5_cycle2_evaluate.py"),
                   "--deterministic-slalom", str(plan_path), str(checkpoint), str(deterministic_dir)]
    try:
        det_child = subprocess.run(det_command, cwd=ROOT, capture_output=True, text=True,
                                   timeout=2 * 60 * 60, check=False)
    except subprocess.TimeoutExpired as error:
        deterministic_dir.mkdir(parents=True, exist_ok=True)
        (deterministic_dir / "worker-timeout.txt").write_text(
            f"timeout_seconds={error.timeout}\ncheckpoint_sha256={metadata['checkpoint_sha256']}\n",
            encoding="utf-8")
        raise RuntimeError("deterministic Slalom evaluation timed out; fine-tuning stopped") from error
    if det_child.returncode != 0:
        deterministic_dir.mkdir(parents=True, exist_ok=True)
        (deterministic_dir / "worker-failure.txt").write_text(
            f"returncode={det_child.returncode}\nstdout={det_child.stdout[-3000:]}\nstderr={det_child.stderr[-5000:]}\n",
            encoding="utf-8")
        raise RuntimeError("deterministic Slalom evaluation failed; fine-tuning stopped")
    det = json.loads((deterministic_dir / "EVALUATION-REPORT.json").read_text(encoding="utf-8"))
    slalom = report["grouped"]["slalom-v1:stochastic"]
    slalom_fixed = sum(row["terminal"]["finished"] for row in report["cases"]
                       if row["track_id"] == "slalom-v1" and row["episode"] < 10)
    slalom_fresh = sum(row["terminal"]["finished"] for row in report["cases"]
                       if row["track_id"] == "slalom-v1" and row["episode"] >= 10)
    others = {track: report["grouped"][f"{track}:stochastic"]["finishes"]
              for track in TRACKS if track != "slalom-v1"}
    det_count = det["grouped"]["slalom-v1:deterministic"]["finishes"]
    eligible = (det_count >= 8 and slalom["finishes"] >= 12 and slalom_fixed >= 8
                and slalom_fresh >= 4 and all(value == 15 for value in others.values()))
    result = {
        "timestep": metadata["timestep"], "checkpoint_sha256": metadata["checkpoint_sha256"],
        "stochastic_slalom": {"combined": slalom["finishes"], "fixed": slalom_fixed,
                              "fresh_development": slalom_fresh},
        "deterministic_slalom": det_count, "other_stochastic_tracks": others,
        "eligible": eligible, "evaluation_report_sha256": report["sha256"],
        "deterministic_report_sha256": det["sha256"],
        "stop_for_material_regression": (slalom["finishes"] < 12 or slalom_fixed < 8
                                         or slalom_fresh < 4 or any(v < 15 for v in others.values())),
    }
    history_path = evidence_root / "checkpoint-selection.jsonl"
    with history_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    return not result["stop_for_material_regression"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", nargs=3, metavar=("EVIDENCE_ROOT", "EVAL_PLAN", "CAMPAIGN_PLAN"))
    parser.add_argument("--train", nargs=2, metavar=("TRAINING_PLAN", "EVIDENCE_ROOT"))
    parser.add_argument("--preflight", nargs=2, metavar=("TRAINING_PLAN", "EVIDENCE_ROOT"))
    args = parser.parse_args()
    if args.freeze:
        evidence_root, eval_path, campaign_path = map(Path, args.freeze)
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation = freeze_evaluation_plan(eval_path, campaign_path)
        training = make_training_plan(ROOT, evidence_root, eval_path)
        evidence_root.mkdir(parents=True, exist_ok=False)
        (evidence_root / "training-plan.json").write_text(
            json.dumps(training, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"training_plan_sha256": training["sha256"],
                          "evaluation_plan_sha256": evaluation["sha256"]}))
    elif args.preflight:
        plan_path, root = map(Path, args.preflight)
        result = run_m5_cycle2(ROOT, plan_path=plan_path, output_directory=root / "preflight",
                               checkpoint_observer=lambda *_: True, dry_run=True)
        print(json.dumps(result, sort_keys=True))
    elif args.train:
        plan_path, root = map(Path, args.train)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("sha256") != canonical_sha256({k: v for k, v in plan.items() if k != "sha256"}):
            raise ValueError("training plan self-hash mismatch")
        try:
            result = run_m5_cycle2(ROOT, plan_path=plan_path, output_directory=root / "training",
                                   checkpoint_observer=lambda checkpoint, meta:
                                   evaluate_after_checkpoint(checkpoint, meta, plan, root))
        except BaseException as error:
            failure = {"status": "BLOCKED_BY_REGRESSION_OR_EVIDENCE",
                       "error_type": type(error).__name__, "error": str(error),
                       "training_plan_sha256": plan["sha256"],
                       "heldout_access": False, "scientific_evidence": False}
            (root / "TRAINING-FAILURE.json").write_text(
                json.dumps(failure, sort_keys=True, indent=2, allow_nan=False) + "\n",
                encoding="utf-8")
            raise
        history_path = root / "checkpoint-selection.jsonl"
        history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()
                   if line.strip()] if history_path.is_file() else []
        eligible = [row for row in history if row["eligible"]]
        if eligible:
            winner = max(eligible, key=lambda row: (
                row["deterministic_slalom"], row["stochastic_slalom"]["combined"],
                row["stochastic_slalom"]["fixed"], row["stochastic_slalom"]["fresh_development"]))
            verdict = "CANDIDATE_ELIGIBLE_FOR_INDEPENDENT_AUDIT"
            checkpoint = root / "training" / f"checkpoint-{winner['timestep']}.zip"
            checkpoint_sha = file_sha256(checkpoint)
            if checkpoint_sha != winner["checkpoint_sha256"]:
                raise ValueError("selected checkpoint hash differs from paired selection evidence")
        else:
            winner = None
            verdict = "FINE_TUNE_TARGET_NOT_MET"
            checkpoint = STD050
            checkpoint_sha = EXPECTED_PARENT_SHA
        selection = {"schema_version": "f1rl_m5_cycle2_selection_v1",
                     "verdict": verdict, "selected_checkpoint_path": str(checkpoint.resolve()),
                     "selected_checkpoint_sha256": checkpoint_sha,
                     "selected_timestep": None if winner is None else winner["timestep"],
                     "eligible_checkpoints": len(eligible), "checkpoint_count": len(history),
                     "history": history, "heldout_access": False,
                     "scientific_evidence": False}
        selection["sha256"] = canonical_sha256(selection)
        (root / "SELECTION-RESULT.json").write_text(
            json.dumps(selection, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (root / "CYCLE2-TRAINING-RESULT.json").write_text(
            json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": verdict, "additional_timesteps": result["additional_timesteps"],
                          "final_num_timesteps": result["final_num_timesteps"],
                          "selected_checkpoint_sha256": checkpoint_sha}))


if __name__ == "__main__":
    main()
