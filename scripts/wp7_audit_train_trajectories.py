"""Audit stored WP7 real PPO runs, without changing original model evidence.

Use --run for calibration and confirmation. Writes a NEW audit file outside
source worktree and does not access HELD-OUT tracks.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path

from f1rl.learning.trajectory_audit import audit_trajectory_run


def main()->int:
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="append",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    audits=[audit_trajectory_run(repository_root=root,run_directory=path) for path in args.run]
    output=args.output.resolve()
    if output.is_relative_to(root):
        parser.error("audit output must remain outside source Git worktree")
    if output.exists():
        parser.error("refuse overwrite of an existing independent audit report")
    output.parent.mkdir(parents=True,exist_ok=True)
    body={"schema_version":"wp7-recovery-dual-seed-trajectory-audit-v1",
          "scientific_evidence":False,"heldout_rollouts":0,"audits":audits}
    payload=(json.dumps(body,indent=2,sort_keys=True,allow_nan=False)+"\n").encode("utf-8")
    with output.open("xb") as stream:
        stream.write(payload)
    print(json.dumps({
        "output":str(output),"sha256":sha256(payload).hexdigest(),
        "scientific_evidence":False,"heldout_rollouts":0,
        "run_summaries":[{
            "run_id":row["run_id"],
            "successes":sum(int(track["success"]) for track in row["results"]),
            "track_count":len(row["results"]),
            "min_xz_gap_m":min((x["min_continuous_xz_point_to_obstacle_m"]
                for x in row["results"] if x["min_continuous_xz_point_to_obstacle_m"] is not None),default=None),
            "max_route_deviation_m":max(x["max_horizontal_route_deviation_m"] for x in row["results"]),
        } for row in audits],
    },indent=2),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
