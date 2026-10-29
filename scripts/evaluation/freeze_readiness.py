"""Verify original 200k WP7 TRAIN evidence and produce fail-closed freeze gate.

Never overwrites historical models/reports or authorizes HELD-OUT.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path



from src.rl.freeze_readiness import build_freeze_readiness_report


def main()->int:
    root=Path(__file__).resolve().parents[2]
    default=Path(r"D:\Projects\.archives\F1-RL\wp7-exploratory")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration",type=Path,
        default=default/"2026-10-11_205037_ppo-multitrack-seed0-sample0")
    parser.add_argument("--confirmation",type=Path,
        default=default/"2026-10-12_203835_ppo-multitrack-seed1-sample1")
    parser.add_argument("--xz-audit",type=Path,
        default=default/"audits/dual-seed-train-trajectory-audit-v1.json")
    parser.add_argument("--swept-audit",type=Path,
        default=default/"audits/dual-seed-swept-3d-body-proxy-v1.json")
    parser.add_argument("--road-union-audit",type=Path,
        default=default/"audits/wp7-right-turn-centerline-corridor-v1.json")
    parser.add_argument("--stochastic-screen",type=Path,
        default=default/"audits/wp7-train-stochastic-action-sensitivity-v1.json")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if output.is_relative_to(root):
        parser.error("readiness output must be outside source Git worktree")
    if output.exists():
        parser.error("refusing to overwrite existing readiness report")
    report=build_freeze_readiness_report(
        repository_root=root,calibration_run=args.calibration,
        confirmation_run=args.confirmation,xz_audit_path=args.xz_audit,
        swept_3d_audit_path=args.swept_audit,
        road_union_audit_path=args.road_union_audit,
        stochastic_screen_path=args.stochastic_screen,
    )
    payload=(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n").encode("utf-8")
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("xb") as f:
        f.write(payload)
        f.flush()
    print(json.dumps({
        "output":str(output),"sha256":sha256(payload).hexdigest(),
        "status":report["status"],
        "verified_runs":len(report["registered_runs"]),
        "linked_audits":len(report["linked_audits"]),
        "blocking_decisions":len(report["blocking_decisions"]),
        "scientific_freeze_permitted":report["scientific_freeze_permitted"],
        "heldout_evaluation_permitted":report["heldout_evaluation_permitted"],
    },indent=2),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
