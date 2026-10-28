"""Read-only current WP3 frozen 3-case physical admission check.

Exit 3 = full evaluation BLOCKED. Never launches any training or Eval.
"""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.validation.wp3_current_eval_gate import (
    require_latest_replay_evidence, require_registered_full_evaluation
)
def main():
    result=require_latest_replay_evidence()
    print(json.dumps(result,sort_keys=True))
    try:require_registered_full_evaluation(result)
    except RuntimeError as error:
        print(str(error),file=sys.stderr)
        return 3
    return 0
if __name__=="__main__":
    raise SystemExit(main())
