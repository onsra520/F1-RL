"""Immutable, source-bound technical registration of WP3 Crash Terminal v1."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.validation.wp3_crash_contract_registration import assemble_registration
import os
HISTORY=Path(os.environ.get("F1RL_WP3_EVIDENCE", r"D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1"))
REPLAY=ROOT/"docs/research/WP3-V4AJ-REGISTERED-CRASH-V1-THREE-REPLAY-20261024.json"
RECEIPT=ROOT/"docs/research/WP3-CRASH-TERMINAL-V1-REGRESSION-20261024.txt"
OUTPUT=ROOT/"docs/research/WP3-CRASH-TERMINAL-V1-TECHNICAL-REGISTRY-20261024.json"
def main():
    x=assemble_registration(ROOT,HISTORY,REPLAY,RECEIPT)
    with OUTPUT.open("x",encoding="utf8",newline="\n") as output:
        json.dump(x,output,indent=2,ensure_ascii=False,allow_nan=False)
        output.write("\n")
    print(json.dumps({
        "status":x["status"],"tests_passed":x["engineering_regression_passed"],
        "frozen_checkpoints":x["registered_original_checkpoint_count"],
        "replay_cases":len(x["same_build_replay_cases"]),
        "contract_sha256":x["engineering_scoring_contract_sha256"],
        "source_bundle_sha256":x["source_bundle_sha256"],
        "registry_record_sha256":x["record_sha256"],
        "scientific_scoring_approved":x["scientific_scoring_approved"],
        "registered_eval_authorized":x["full_5760_episode_evaluation_authorized"]
    },sort_keys=True))
if __name__=="__main__":main()
