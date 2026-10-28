"""Read-only WP7 3D swept body-proxy sensitivity audit for stored TRAIN PPO runs."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import sys as _entry_sys
from pathlib import Path as _EntryPath

_entry_root = str(_EntryPath(__file__).resolve().parents[1])
if _entry_root not in _entry_sys.path:
    _entry_sys.path.insert(0, _entry_root)

from src.rl.swept_volume_audit import audit_3d_swept_run


def main()->int:
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="append",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if output.is_relative_to(root):
        parser.error("report output must be outside the Git source worktree")
    if output.exists():
        parser.error("existing audit report cannot be overwritten")
    if len(set(path.resolve(strict=True) for path in args.run))!=len(args.run):
        parser.error("duplicate run directories are disallowed")
    audits=[audit_3d_swept_run(repository_root=root,run_directory=path) for path in args.run]
    result={
        "schema_version":"wp7-recovery-dual-seed-3d-body-proxy-audit-v1",
        "scientific_evidence":False,
        "heldout_rollouts":0,
        "audits":audits,
    }
    raw=(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n").encode("utf-8")
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
    print(json.dumps({
        "output":str(output),
        "sha256":sha256(raw).hexdigest(),
        "scientific_evidence":False,
        "run_summaries":[{
            "run_id":audit["run_id"],
            "successes":sum(int(x["success"]) for x in audit["track_results"]),
            "track_count":len(audit["track_results"]),
            "profiles":[{
                "name":name,
                "min_conservative_gap_m":min(
                    (track["profiles"][i]["min_conservative_gap_m"] for track in audit["track_results"]
                     if track["profiles"][i]["min_conservative_gap_m"] is not None),
                    default=None,
                ),
                "all_obstacle_tracks_pass":all(
                    track["profiles"][i]["conservative_clearance_pass"] is not False
                    for track in audit["track_results"]
                ),
            } for i,name in enumerate(p["name"] for p in audit["track_results"][0]["profiles"])],
        } for audit in audits],
    },indent=2),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
