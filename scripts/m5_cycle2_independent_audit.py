"""Read-only integrity audit for M5 Cycle 2 paired evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)


def audit(report_path: Path, plan_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    report_digest = report.get("sha256")
    if report_digest != canonical_sha256({k: v for k, v in report.items() if k != "sha256"}):
        raise ValueError("evaluation report canonical SHA-256 mismatch")
    checkpoint_digest = file_sha256(checkpoint_path)
    if report.get("checkpoint_sha256") != checkpoint_digest:
        raise ValueError("checkpoint SHA-256 mismatch")
    if report.get("plan_sha256") != plan.get("sha256"):
        raise ValueError("evaluation plan SHA-256 mismatch")
    if plan.get("heldout_access") is not False or report.get("heldout_access") is not False:
        raise ValueError("HELD-OUT access flag is not false")

    expected: dict[tuple[str, str, int], tuple[int, int]] = {}
    for track, roster in plan["seed_roster"].items():
        for row in roster:
            expected[(track, "stochastic", int(row["episode"]))] = (
                int(row["environment_seed"]), int(row["policy_seed"]))
    for row in plan["seed_roster"]["slalom-v1"][:10]:
        expected[("slalom-v1", "deterministic", int(row["episode"]))] = (
            int(row["environment_seed"]), int(row["policy_seed"]))

    cases = report.get("cases", [])
    seen: set[tuple[str, str, int]] = set()
    group_acc: dict[str, dict[str, Any]] = {}
    raw_hash_count = 0
    normalized_hash_count = 0
    for case in cases:
        track = str(case["track_id"])
        mode = str(case["mode"])
        episode = int(case["episode"])
        key = (track, mode, episode)
        if key in seen or key not in expected:
            raise ValueError(f"duplicate or unregistered roster case: {key}")
        seen.add(key)
        if expected[key] != (int(case["environment_seed"]), int(case["policy_seed"])):
            raise ValueError(f"seed mismatch: {key}")
        trace_path = Path(case["trace_path"])
        raw_digest = file_sha256(trace_path)
        if raw_digest != case["trace_sha256"]:
            raise ValueError(f"raw trace SHA-256 mismatch: {key}")
        lines = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        if not lines or len(lines) != int(case["step_count"]):
            raise ValueError(f"trace row count mismatch: {key}")
        if [int(row["step"]) for row in lines] != list(range(1, len(lines) + 1)):
            raise ValueError(f"trace steps are not contiguous: {key}")
        if any(row.get("physics_sha256") != case.get("physics_sha256")
               for row in lines if "physics_sha256" in row):
            raise ValueError(f"trace Physics SHA mismatch: {key}")

        applied = [row["applied_action"] for row in lines]
        speed = [float(row["speed_mps"]) for row in lines]
        action_summary = case["action_summary"]
        for action_name in ("steering", "throttle", "brake"):
            mean = sum(float(action[action_name]) for action in applied) / len(applied)
            if not close(mean, float(action_summary[f"mean_{action_name}"])):
                raise ValueError(f"{action_name} aggregate mismatch: {key}")
        if not close(sum(speed) / len(speed), float(action_summary["mean_speed_mps"])):
            raise ValueError(f"mean speed mismatch: {key}")
        if not close(sum(speed) / len(speed), float(case["metrics"]["mean_speed"])):
            raise ValueError(f"metrics mean speed mismatch: {key}")
        if not close(max(speed), float(case["metrics"]["max_speed"])):
            raise ValueError(f"metrics max speed mismatch: {key}")

        last = lines[-1]
        terminal = case["terminal"]
        metrics = case["metrics"]
        if bool(last["finished"]) != bool(terminal["finished"]):
            raise ValueError(f"finish state mismatch: {key}")
        if bool(last["terminated"]) != bool(case["terminal"]["finished"] or terminal["reason"] == "physical_crash"):
            raise ValueError(f"terminal state mismatch: {key}")
        if not close(float(metrics["final_normalized_route_progress"]),
                     float(last.get("normalized_route_progress", metrics["final_normalized_route_progress"]))):
            raise ValueError(f"route progress mismatch: {key}")
        if int(metrics["steps"]) != len(lines):
            raise ValueError(f"metric step count mismatch: {key}")
        if bool(metrics["success"]) != bool(terminal["finished"]):
            raise ValueError(f"success mismatch: {key}")

        group_key = f"{track}:{mode}"
        group = group_acc.setdefault(group_key, {"episodes": 0, "finishes": 0,
                                                  "crashes": 0, "step_limits": 0,
                                                  "progress_sum": 0.0})
        group["episodes"] += 1
        group["finishes"] += int(bool(metrics["success"]))
        group["crashes"] += int(terminal.get("reason") == "physical_crash")
        group["step_limits"] += int(bool(metrics.get("truncated")))
        group["progress_sum"] += float(metrics["final_normalized_route_progress"])
        raw_hash_count += 1
        normalized_hash_count += int(bool(case.get("trace_sha256")))

    if seen != set(expected):
        missing = sorted(set(expected) - seen)
        raise ValueError(f"evaluation cases missing from roster: {missing[:5]}")
    recomputed: dict[str, dict[str, Any]] = {}
    for key, group in group_acc.items():
        count = group["episodes"]
        recomputed[key] = {
            "episodes": count,
            "finishes": group["finishes"],
            "crashes": group["crashes"],
            "step_limits": group["step_limits"],
            "mean_progress": group["progress_sum"] / count,
        }
    reported_groups = report.get("grouped", {})
    stochastic_recomputed = {key: value for key, value in recomputed.items()
                             if key.endswith(":stochastic")}
    for key, actual in reported_groups.items():
        computed = stochastic_recomputed.get(key)
        if computed is None:
            raise ValueError(f"unrecomputed group: {key}")
        for field in ("episodes", "finishes", "crashes", "step_limits"):
            if int(actual[field]) != computed[field]:
                raise ValueError(f"group {field} mismatch: {key}")
        if not close(float(actual["mean_progress"]), computed["mean_progress"]):
            raise ValueError(f"group mean progress mismatch: {key}")
    if set(reported_groups) != set(stochastic_recomputed):
        raise ValueError("reported and recomputed stochastic group keys differ")
    deterministic_count = sum(value["episodes"] for key, value in recomputed.items()
                              if key.endswith(":deterministic"))
    if deterministic_count != 10:
        raise ValueError(f"expected 10 deterministic Slalom cases, got {deterministic_count}")

    return {
        "schema_version": "f1rl_m5_cycle2_independent_audit_v1",
        "report_path": str(report_path),
        "report_sha256": report_digest,
        "plan_sha256": plan["sha256"],
        "checkpoint_sha256": checkpoint_digest,
        "cases_verified": len(cases),
        "raw_trace_sha256_verified": raw_hash_count,
        "trace_hash_fields_verified": normalized_hash_count,
        "deterministic_slalom_cases": deterministic_count,
        "groups_recomputed": recomputed,
        "heldout_access": False,
        "scientific_evidence": False,
        "verdict": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.report, args.plan, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "cases_verified": result["cases_verified"],
                      "report_sha256": result["report_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
