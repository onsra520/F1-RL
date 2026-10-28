"""Verify WP3 Crash Terminal v1 engineering registry against current source/seed."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from f1rl.research.wp3_crash_contract_registration import assemble_registration
from f1rl.learning.crash_terminal_registered import _sha
import os
HISTORY=Path(os.environ.get("F1RL_WP3_EVIDENCE", r"D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1"))
REPLAY=ROOT/"docs/research/WP3-V4AJ-REGISTERED-CRASH-V1-THREE-REPLAY-20261024.json"
TESTS=ROOT/"docs/research/WP3-CRASH-TERMINAL-V1-REGRESSION-20261024.txt"
REGISTRY=ROOT/"docs/research/WP3-CRASH-TERMINAL-V1-TECHNICAL-REGISTRY-20261024.json"
def main():
    reference=json.loads(REGISTRY.read_text("utf8"))
    if reference["record_sha256"]!=_sha({k:v for k,v in reference.items() if k!="record_sha256"}):
        raise ValueError("registry record checksum invalid")
    observed=assemble_registration(ROOT,HISTORY,REPLAY,TESTS)
    observed["created_at_utc"]=reference["created_at_utc"]
    observed["record_sha256"]=_sha({k:v for k,v in observed.items() if k!="record_sha256"})
    if observed!=reference:
        raise ValueError("registered crash source/roster/model/replay/test changed")
    assert not reference["scientific_scoring_approved"]
    assert not reference["full_5760_episode_evaluation_authorized"]
    print(json.dumps({"registration":"ENGINEERING_CRASH_V1_SOURCE_VERIFIED",
        "engineering_regression":reference["engineering_regression_passed"],
        "replay_cases":len(reference["same_build_replay_cases"]),
        "source_bundle_sha256":reference["source_bundle_sha256"],
        "owner_signoff":reference["owner_scientific_signoff"],
        "full_eval_authorized":reference["full_5760_episode_evaluation_authorized"]},
        sort_keys=True))
if __name__=="__main__":main()
