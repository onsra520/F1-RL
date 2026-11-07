"""One-shot source-bound six-case frozen PPO smoke, separate from TRAIN-5 draft.

Never trains, allocates the 18 reserved training seeds, loads HELD-OUT, or
changes any scientific/source asset. A user-authorized versioned smoke protocol
is required; the historical TRAIN-5 draft admission guards remain closed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/research/wp7-train5-frozen-ppo-smoke-v1.json"
ANCHOR = "bb2d48bc17ba1e5bef75764998977fdc0cef27f0"
BRANCH = "handoff/physics-v4-khanh-bang-20261106"
EVIDENCE = Path(r"D:\F1-RL-PHYSICS-V4-EVIDENCE-20261106")
EXPECTED_MODELS = ("C-seed0", "B-seed1", "A-seed2")
EXPECTED_TRACKS = ("right-turn-train-v2", "left-turn-train-v2")
LIMIT = 2500
SEED = 20261106


class SmokeBlocked(RuntimeError):
    pass


def sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as inp:
        for block in iter(lambda: inp.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(ROOT), *args), check=True,
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def require(cond: bool, reason: str) -> None:
    if not cond:
        raise SmokeBlocked(reason)


def load_protocol() -> tuple[dict, dict]:
    from scripts.wp7_train5_handoff_preflight import audit
    p = json.loads(PROTOCOL.read_text(encoding="utf8"))
    require(p.get("schema") == "f1rl_train5_immutable_six_case_smoke_v1", "wrong smoke schema")
    require(p.get("status") == "APPROVED_BY_OWNER_FOR_FROZEN_PPO_SMOKE_ONLY", "no owner approval")
    require(p.get("frozen_source_base_commit") == ANCHOR, "wrong source anchor")
    require(p.get("branch") == BRANCH, "wrong branch")
    require(tuple(p.get("model_ids", ())) == EXPECTED_MODELS, "checkpoint set drift")
    require(tuple(p.get("track_ids", ())) == EXPECTED_TRACKS, "TRAIN turn set drift")
    require(p.get("reset_seed") == SEED and p.get("max_steps_per_case") == LIMIT, "smoke cap/seed drift")
    require(p.get("required_case_count") == 6 and p.get("one_shot") is True, "case count/one-shot drift")
    require(p.get("deterministic_prediction") is True and p.get("device") == "cpu", "determinism drift")
    for flag in (
        "reuse_training_seeds", "read_heldout_geometry", "train_ppo",
        "allow_training_after_smoke",
    ):
        require(p.get(flag) is False, "unsafe smoke permission: " + flag)
    require(p.get("separate_training_protocol_required") is True, "no independent training gate")
    require(p.get("allowed_output_root") == str(EVIDENCE), "unexpected output path")
    require(git("branch", "--show-current") == BRANCH, "wrong local branch")
    require(git("merge-base", "--is-ancestor", ANCHOR, "HEAD") == "", "source anchor not ancestor")
    require(not git("status", "--porcelain"), "source working tree not clean")
    source = audit(ROOT, EVIDENCE)
    require(source["result"] == "PASS_SOURCE_LOCK_NO_TRAINING", "44 file lock rejected")
    require(source["external_witness_count"] == 4, "Step3 sources missing")
    registry = json.loads((ROOT / p["seed_registry"]).read_text("utf8"))
    require(registry.get("registry_gate_status") ==
            "CLOSED_WITH_OWNER_ACCEPTED_EVIDENCE_EXCEPTION", "global registry not accepted")
    for gate in ("ppo_smoke_permission", "ppo_training_permission",
                 "heldout_policy_evaluation_permission", "scientific_release_permission"):
        require(registry.get(gate) is False, "unexpected prior permission in registry: " + gate)
    draft = json.loads((ROOT / p["draft_protocol"]).read_text("utf8"))
    require(draft.get("status") == "draft_not_authorized", "frozen draft modified")
    require(draft.get("training_permission") is False and
            draft.get("smoke_permission") is False, "frozen draft guards modified")
    selected = set(p["track_ids"])
    require(selected <= set(draft["train_ids"]), "requested non-TRAIN track")
    require(selected.isdisjoint(set(draft["heldout_ids"])), "HELD-OUT leak")
    for model in EXPECTED_MODELS:
        filename = f"models/{model}/model-final.zip"
        require(sha(ROOT / filename) == draft["source_sha256"][filename],
                f"checkpoint hash drift: {model}")
    return p, source


def classify(info: dict, terminated: bool, truncated: bool) -> str | None:
    from f1rl.learning.crash_terminal_registered import (
        validate_registered_crash_terminal_info,
    )
    if info.get("finished"):
        return "finished"
    reason = info.get("research_outcome") or {}
    race = info.get("race") or {}
    reason = reason.get("reason") or race.get("terminal_reason")
    if reason == "physical_crash" or info.get("physical_crash_terminal") is not None:
        return ("certified_physical_crash"
                if validate_registered_crash_terminal_info(info)
                else "uncertified_physical_crash")
    if reason == "hard_route_violation":
        return "valid_hard_route_violation"
    if truncated and (reason == "step_limit" or reason is None):
        return "valid_step_limit"
    if terminated or truncated:
        return "runtime_or_contract_exception"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true",
                        help="Validate source, models, gates without running simulation.")
    parser.add_argument("--execute-approved-six", action="store_true",
                        help="Consumes the one-shot smoke slot and runs exactly six cases.")
    args = parser.parse_args()
    if args.execute_approved_six == args.preflight_only:
        parser.error("choose exactly one execution mode")
    p, source = load_protocol()
    if args.preflight_only:
        print(json.dumps({
            "result": "PASS_PREPARE_ONLY",
            "source": git("rev-parse", "HEAD"),
            "six_cases": len(p["model_ids"]) * len(p["track_ids"]),
            "frozen_model_hashes_verified": 3,
            "physics_step3_receipts_verified": source["external_witness_count"],
            "episodes_executed": 0,
            "training_permission": False,
        }, sort_keys=True))
        return 0

    import numpy as np
    import torch
    from f1rl.learning.algorithms.ppo import PPOModel
    from f1rl.research.wp7_train5_v2_protocol import preview_train5_v2
    torch.set_num_threads(1)
    from f1rl.research.wp3_eval_only_coordinator import effective_runtime_profile

    scenarios = preview_train5_v2(
        ROOT,
        registration_file=ROOT / p["draft_protocol"],
        evidence_root=EVIDENCE,
    )
    require(set(p["track_ids"]) <= {s.track_id for s in scenarios.scenarios},
            "TRAIN-only preview roster mismatch")
    envs = {s.track_id: s.factory for s in scenarios.scenarios if s.track_id in p["track_ids"]}
    for track_id, factory in envs.items():
        require(factory.requires_scientific_admission, "not a research Physics factory")
        require(factory.reset_config is None, "unexpected train reset activation")
        require(factory.physics_config.ground_motion_version ==
                p["runtime_profile"]["physics_version"], "physics drift")
        require(factory.race_config.version == p["runtime_profile"]["race_version"],
                "race drift")

    source_commit = git("rev-parse", "HEAD")
    destination = EVIDENCE / (
        "HANDOFF-TRAIN5-PPO-SMOKE-V1-" + source_commit[:10].upper() + "-20261107"
    )
    require(not destination.exists(), "one-shot smoke already attempted: " + str(destination))
    destination.mkdir(parents=False, exist_ok=False)
    marker = destination / "ONE_SHOT_ATTEMPT.json"
    marker.write_text(json.dumps({
        "status": "ATTEMPT_CONSUMED_NOT_EVIDENCE_OF_SUCCESS",
        "source_commit": source_commit,
        "smoke_protocol_sha256": sha(PROTOCOL),
        "source_preflight": source,
        "authorized_cases": [
            [model, track] for model in EXPECTED_MODELS for track in EXPECTED_TRACKS
        ],
        "training_seeds_consumed": [],
        "training_permitted": False,
    }, indent=2) + "\n", encoding="utf8")

    results = []
    expected = set(p["allowed_outcomes"])
    model_hashes = {}
    try:
        for model_id in EXPECTED_MODELS:
            model_path = ROOT / "models" / model_id / "model-final.zip"
            model_hashes[model_id] = sha(model_path)
            model = PPOModel.load(model_path, device="cpu")
            for track in EXPECTED_TRACKS:
                logpath = destination / f"{model_id}__{track}.jsonl"
                env = envs[track].create()
                state = "runtime_or_contract_exception"
                step_count = 0
                checkpoint_ids: list[str] = []
                last_info = {}
                try:
                    with logpath.open("x", encoding="utf8", newline="\n") as stream:
                        stream.write(json.dumps({
                            "type": "start", "model": model_id, "track": track,
                            "source_commit": source_commit, "seed": SEED,
                            "model_sha256": model_hashes[model_id],
                            "deterministic": True, "step_limit": LIMIT,
                        }, sort_keys=True) + "\n")
                        obs, reset_info = env.reset(seed=SEED)
                        require(env.observation_space.contains(obs), "bad reset observation")
                        for step in range(1, LIMIT + 1):
                            action = np.asarray(model.predict(obs, deterministic=True), dtype=np.float32)
                            require(action.shape == (3,) and np.all(np.isfinite(action)),
                                    "invalid PPO action")
                            require(env.action_space.contains(action), "action violates action-space")
                            next_obs, reward, terminated, truncated, info = env.step(action)
                            require(env.observation_space.contains(next_obs), "invalid next observation")
                            require(np.isfinite(float(reward)), "non-finite reward")
                            step_count = step
                            last_info = info
                            checkpoint = info.get("completed_checkpoint_id")
                            if checkpoint is not None:
                                checkpoint_ids.append(checkpoint)
                            state = classify(info, bool(terminated), bool(truncated)) or "running"
                            telemetry = info.get("telemetry") or {}
                            stream.write(json.dumps({
                                "type": "step", "step": step,
                                "action": [float(x) for x in action],
                                "reward": float(reward),
                                "finished": bool(info.get("finished")),
                                "checkpoint": checkpoint,
                                "position": telemetry.get("vehicle_position"),
                                "speed": telemetry.get("vehicle_speed"),
                                "terminal": state if terminated or truncated else None,
                            }, default=str, allow_nan=False) + "\n")
                            obs = next_obs
                            if terminated or truncated:
                                break
                        else:
                            state = "valid_step_limit"
                        stream.write(json.dumps({
                            "type": "case_complete", "model": model_id, "track": track,
                            "steps": step_count, "classification": state,
                            "checkpoints": checkpoint_ids,
                            "research_outcome": last_info.get("research_outcome"),
                        }, default=str, allow_nan=False) + "\n")
                finally:
                    env.close()
                result = {
                    "model_id": model_id, "track_id": track, "reset_seed": SEED,
                    "steps": step_count, "classification": state,
                    "checkpoints": checkpoint_ids,
                    "log_sha256": sha(logpath),
                    "terminal_reason": (last_info.get("research_outcome") or {}).get("reason"),
                }
                results.append(result)
                print(json.dumps({"completed_case": len(results), "out_of": 6, **result}, sort_keys=True), flush=True)
                require(state in expected, "scientific smoke fails closed on: " + state)
        require(len(results) == 6, "fewer than six cases ran")
        require(all(sha(ROOT / "models" / m / "model-final.zip") == digest
                    for m, digest in model_hashes.items()), "model changed during inference")
        counts = dict(Counter(row["classification"] for row in results))
        status = "PASS_SIX_VALID_EXECUTIONS_NOT_POLICY_QUALITY"
        summary = {
            "result": status, "source_commit": source_commit,
            "registry_status": "CLOSED_WITH_OWNER_ACCEPTED_EVIDENCE_EXCEPTION",
            "smoke_protocol_sha256": sha(PROTOCOL),
            "case_count": len(results), "outcomes": counts, "cases": results,
            "model_hashes": model_hashes,
            "training_seeds_consumed": [], "train_ppo_executed": False,
            "heldout_geometry_read": False,
            "training_allowed": False,
            "model_quality_claim": "NONE_SMOKE_ONLY",
        }
        (destination / "smoke-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf8"
        )
        print(json.dumps({"smoke_result": status, "summary": str(destination / "smoke-summary.json")}), flush=True)
        return 0
    except BaseException as exc:
        (destination / "SMOKE_FAILED_NO_RETRY.json").write_text(json.dumps({
            "result": "FAIL_CLOSED_ONE_SHOT_CONSUMED",
            "exception": type(exc).__name__ + ": " + str(exc),
            "traceback": traceback.format_exc(limit=12),
            "completed_cases": results,
            "training_allowed": False,
        }, indent=2, default=str) + "\n", encoding="utf8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
