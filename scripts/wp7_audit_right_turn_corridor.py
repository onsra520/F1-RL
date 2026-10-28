"""Replay only the archived Right Turn TRAIN final PPO models for road-union audit."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from src.learning.corridor_audit import audit_right_turn_corridor


def main()->int:
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,action="append",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if output.is_relative_to(root):
        parser.error("report must be written outside Git worktree")
    resolved=[x.resolve(strict=True) for x in args.run]
    if len(resolved)!=2 or len(set(resolved))!=2:
        parser.error("exactly two distinct calibration and confirmation runs are required")
    if output.exists():
        parser.error("read-only audit refuses overwrite")
    rows=[audit_right_turn_corridor(repository_root=root,run_directory=path) for path in resolved]
    report={
        "schema_version":"wp7-recovery-dual-seed-right-turn-road-union-audit-v1",
        "scientific_evidence":False,
        "heldout_rollouts":0,
        "audits":rows,
    }
    raw=(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n").encode("utf-8")
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({
        "output":str(output),"sha256":sha256(raw).hexdigest(),
        "status":"diagnostic_requires_approval","heldout_rollouts":0,
        "runs":[{
            "run_id":x["run_id"],
            "centerline_uncovered_road_path_xz_m":x["centerline_uncovered_road_path_xz_m"],
            "max_sampled_route_deviation_m":x["max_sampled_route_deviation_m"],
            "upper_bound_route_deviation_between_samples_m":x["upper_bound_route_deviation_between_samples_m"],
            "success":x["success"],
        } for x in rows],
    },indent=2),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
