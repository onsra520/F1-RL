"""Re-verify source-bound physics v4 engineering evidence without overwriting."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from f1rl.research.wp3_v4_technical_acceptance import (
    _json,hash_manifest,source_bound_acceptance
)
RECORD=ROOT/"docs/research/WP3-PHYSICS-V4-TECHNICAL-ACCEPTANCE-20261024.json"
REPLAY=ROOT/"docs/research/WP3-V4AH-PHYSICS-V4-ACCEPTANCE-REPLAY-20261024.json"
RECEIPT=ROOT/"docs/research/WP3-V4-PHYSICS-ENGINEERING-PYTEST-20261024.txt"
HISTORY=Path(r"D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1")

def main()->None:
    saved=_json(RECORD)
    original={k:v for k,v in saved.items() if k!="record_sha256"}
    if hash_manifest(original)!=saved.get("record_sha256"):
        raise ValueError("acceptance record was edited or malformed")
    current=source_bound_acceptance(ROOT,HISTORY,REPLAY,test_receipt_file=RECEIPT)
    current.pop("record_sha256")
    current["created_at_utc"]=saved["created_at_utc"]
    if current!=original:
        raise ValueError("source, checkpoint, roster, replay, pytest, or environment changed")
    assert saved["overall_status"]=="TECHNICAL_GATE_PASS_SCIENTIFIC_SIGNOFF_PENDING"
    assert saved["owner_scientific_physics_signoff"] is False
    assert saved["production_and_registered_eval_admitted"] is False
    print(json.dumps({"result":"SOURCE_BOUND_ENGINEERING_RECEIPT_VERIFIED",
                      "owner_scientific_signoff":False,
                      "full_eval_authorized":False,
                      "physical_source_sha256":saved["physics_source_bundle_sha256"],
                      "registered_final_checkpoints":saved["registered_final_model_count"],
                      "replays":len(saved["three_exact_replays"]),
                      "regression_pass":saved["focused_physics_evaluator_tests_passed"]},
                     sort_keys=True))

if __name__=="__main__":main()
