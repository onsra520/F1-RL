"""Strict Issue #51 Eval-only frozen input and source gate. No episode execution."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from f1rl.research.wp3_eval_only_admission import build_eval_only_preflight,refuse_full_eval
HISTORY=Path(r"D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1")
REPLAY=ROOT/"docs/research/WP3-CE0E55D-THREE-REPLAY-PREFLIGHT-20261024.json"
OUTPUT=ROOT/"docs/research/WP3-EVAL-ONLY-COMPLETION-GATE-20261024.json"
def main():
    check=build_eval_only_preflight(ROOT,HISTORY,REPLAY)
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    with OUTPUT.open("x",encoding="utf8",newline="\n") as dest:
        json.dump(check,dest,indent=2,ensure_ascii=False,allow_nan=False)
        dest.write("\n")
    try:refuse_full_eval(check)
    except RuntimeError as exc:
        assert str(exc).startswith("FULL_EVAL_BLOCKED:")
    else:raise AssertionError("incomplete physics allowed forbidden full Eval")
    print(json.dumps({"status":"BLOCKED","all_12_frozen_models_verified":True,
        "registered_roster_sha256":check["historical_roster_sha256"],
        "cells":check["expected_cells"],"episode_slots":check["expected_episodes"],
        "failed_mandatory_replays":len(check["mandatory_replay_cases"]),
        "registered_eval_started":check["registered_eval_started"],
        "new_episodes_executed":check["new_episodes_executed_by_preflight"],
        "preflight_sha256":check["sha256"]},sort_keys=True))
if __name__=="__main__":main()
