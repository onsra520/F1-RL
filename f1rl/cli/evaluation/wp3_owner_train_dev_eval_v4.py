"""WP3 owner-authorized internal TRAIN-development full evaluation (not scientific).

Windows examples:
  python -m scripts.evaluation.wp3_owner_train_dev_eval_v4 prepare --out D:/new-eval-root --run-id WP3-OWNER-TRAINDEV-20261024-001 --owner-approved
  python -m scripts.evaluation.wp3_owner_train_dev_eval_v4 status --out D:/new-eval-root
  python -m scripts.evaluation.wp3_owner_train_dev_eval_v4 run --out D:/new-eval-root --workers 2
  python -m scripts.evaluation.wp3_owner_train_dev_eval_v4 resume --out D:/new-eval-root --workers 2

Never use this command to change the scientific full gate. Independent
physics certification remains pending and all outputs are development-only.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
from f1rl.research.wp3_owner_train_dev_eval import new_run, load_run, run, snapshot

HISTORY=Path(r"D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1")
OWNER_APPROVAL="I approve the internal TRAIN-development evaluation"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","status","run","resume"))
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--history",type=Path,default=HISTORY)
    parser.add_argument("--run-id")
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--owner-approved",action="store_true",
                        help="Record explicit owner approval, NOT independent scientific signoff")
    args=parser.parse_args()
    if args.command=="prepare":
        if not args.owner_approved:
            parser.error("owner approval explicitly required")
        manifest=new_run(ROOT,args.history,args.out,args.run_id,OWNER_APPROVAL)
        print(json.dumps({"status":"OWNER_TRAIN_DEVELOPMENT_READY",
                          "run_id":manifest["run_id"],
                          "source":manifest["source_head"],
                          "expected":5760,"cells":96,
                          "manifest_sha256":manifest["sha256"],
                          "scientific_gate":"BLOCKED_INDEPENDENT_SIGNOFF_PENDING"},
                         sort_keys=True))
        return 0
    if args.command=="status":
        summary=snapshot(ROOT,args.history,args.out)
        print(json.dumps({k:v for k,v in summary.items() if k!="cell_status"},
                         sort_keys=True))
        return 0
    result=run(ROOT,args.history,args.out,args.workers)
    print(json.dumps({k:v for k,v in result.items() if k!="cell_status"},sort_keys=True))
    return 0 if result["all_cells_complete"] else 2


if __name__=="__main__":
    raise SystemExit(main())
