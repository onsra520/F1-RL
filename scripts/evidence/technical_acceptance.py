"""Create source-bound v4 technical acceptance evidence (owner review still pending)."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2]
from src.validation.wp3_v4_technical_acceptance import source_bound_acceptance
import os
HISTORY=Path(os.environ.get("F1RL_WP3_EVIDENCE", r"D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1"))
REPLAY=ROOT/"docs/research/WP3-V4AH-PHYSICS-V4-ACCEPTANCE-REPLAY-20261024.json"
TESTS=ROOT/"docs/research/WP3-V4-PHYSICS-ENGINEERING-PYTEST-20261024.txt"
OUTPUT=ROOT/"docs/research/WP3-PHYSICS-V4-TECHNICAL-ACCEPTANCE-20261024.json"

def main():
    candidate=source_bound_acceptance(ROOT,HISTORY,REPLAY,test_receipt_file=TESTS)
    if OUTPUT.exists():raise FileExistsError("acceptance evidence cannot be overwritten")
    with OUTPUT.open("x",encoding="utf8",newline="\n") as writer:
        json.dump(candidate,writer,indent=2,ensure_ascii=False,allow_nan=False)
        writer.write("\n")
    print("STATUS",candidate["overall_status"])
    print("TECHNICAL_TESTS_PASS",candidate["focused_physics_evaluator_tests_passed"])
    print("REPLAY_COMPLETE",len(candidate["three_exact_replays"]))
    print("MODEL_FINAL_SHA_VERIFIED",candidate["registered_final_model_count"])
    print("SOURCE_BUNDLE_SHA256",candidate["physics_source_bundle_sha256"])
    print("ACCEPTANCE_RECORD_SHA256",candidate["record_sha256"])
    print("INDEPENDENT_SIGNATURE",candidate["independent_reviewer_signature"])
    print("REGISTERED_EVAL_ALLOWED",candidate["production_and_registered_eval_admitted"])

if __name__=="__main__":main()
