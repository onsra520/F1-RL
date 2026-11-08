"""Fail-closed admission for the three *current* WP3 physics blockers.

Read-only source-bound verification of the independent scratch diagnostic rows.
No class of invalid/uncertified physical outcome becomes a scored crash here.
This is NOT a science acceptance signature or a 5,760 episode coordinator.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json

CASES={
 ("C-seed0","right-turn-v1","nominal",0):
  ("SuspensionResolutionError","contact"),
 ("A-seed1","ramp-v1","randomized_train",17):
  ("SuspensionResolutionError","bump"),
 ("C-seed0","slalom-v1","nominal",2):
  ("GroundedBodyCollisionUnscored","obstacle"),
}
ROOT=Path(__file__).resolve().parents[2]
ARTIFACTS={
 "right_turn":ROOT/"docs/research/WP3-V4L-RIGHT-TURN-PATCH-BRACKET-20261024.json",
 "ramp":ROOT/"docs/research/WP3-V4N-RAMP-BUMP-TRACE-20261024.json",
 "slalom":ROOT/"docs/research/WP3-V4M-SLALOM-GROUNDED-CONTACT-20261024.json",
}
def _read(path:Path)->dict:
    return json.loads(path.read_text("utf8"))
def check_snapshot()->dict:
    observations=[]
    for name,path in ARTIFACTS.items():
        if not path.is_file():raise ValueError(f"missing required independent replay {name}")
        raw=path.read_bytes()
        obj=json.loads(raw)
        rows=obj.get("cases")
        if type(rows) is not list or len(rows)!=1:
            raise ValueError("required frozen replay must have exactly one case")
        row=rows[0]
        key=(row["model_id"],row["track"],row["stratum"],row["episode"])
        expected=CASES.get(key)
        if expected is None or row.get("error_type")!=expected[0]:
            raise ValueError("frozen case or physical error unexpectedly changed")
        if expected[1] not in row["error"].lower():
            raise ValueError("physics error witness missing domain detail")
        if row.get("scientific_evidence") is not False or obj.get("full_5760_eval_admission")!="BLOCKED":
            raise ValueError("unregistered evidence falsely promoted to official scoring")
        observations.append({"case":list(key),"error_type":row["error_type"],
            "witness":row["error"],"file":path.name,
            "sha256":sha256(raw).hexdigest(),"certified_valid":False})
    if len({tuple(x["case"]) for x in observations})!=3:
        raise ValueError("all three independent frozen replay fixtures required")
    return {
       "schema_version":"wp3-physics-frozen-three-replay-admission-v1",
       "status":"BLOCKED_UNCERTIFIED_PHYSICS",
       "observed_invalid_scratch_cases":observations,
       "physics_profile":"spatial_road_edge_partial_lifecycle_v4_DEV_ONLY",
       "replay_cases_certified":0,
       "replay_cases_required":3,
       "new_eval_episodes":0,
       "old_eval_evidence_modified":False,
       "registered_eval_started":False,
       "scientific_evidence":False,
       "release_allowed":False,
       "reason":"Road contact witness lacks certified event impulse, ramp penetrates bump hard-stop, obstacle hit is unscored: no registered owner-certified crash terminal."
    }

def demand_full_evaluation_authorization(report:dict)->None:
    if report != check_snapshot() or report["status"]!="PASS":
        raise RuntimeError("FULL_5760_EVAL_BLOCKED: three source-bound physical replay gates not certified")
    raise RuntimeError("FULL_5760_EVAL_BLOCKED: independent evaluator registration required")
