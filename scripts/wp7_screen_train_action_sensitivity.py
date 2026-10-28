"""Execute registered stochastic-action sensitivity screen on TWO archived TRAIN policies."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import torch

from src.learning.robustness_screen import screen_saved_train_policy
from src.learning.geometry_review import load_geometry_review


@contextmanager
def action_rng_scope(seed:int):
    # Experimental CLI, not a direct production-module torch dependency.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


def main()->int:
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration",type=Path,required=True)
    p.add_argument("--confirmation",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    runs=[a.calibration.resolve(strict=True),a.confirmation.resolve(strict=True)]
    if runs[0]==runs[1]:
        p.error("calibration and confirmation must be different")
    if ("seed0-sample0" not in runs[0].name or "seed1-sample1" not in runs[1].name):
        p.error("registered source seed order changed")
    path=a.output.resolve()
    if path.exists() or path.is_relative_to(root) or any(path.is_relative_to(run) for run in runs):
        p.error("refuse overwrite, worktree output or overwrite inside original runs")
    geometry=load_geometry_review(root/"configs/recovery/wp7-vehicle-geometry-review.yml")
    screens=[]
    for run in runs:
        item=screen_saved_train_policy(repository_root=root,run_directory=run,rng_scope=action_rng_scope)
        screens.append(item)
        print(json.dumps({"run_id":item["run_id"],
            "track_results":[(x["track_id"],x["success_count"],x["episode_count"])
                for x in item["train_tracks"]]},sort_keys=True),flush=True)
    report={
        "schema_version":"wp7-recovery-dual-seed-action-sensitivity-v1",
        "status":"exploratory_not_scientific",
        "scientific_evidence":False,
        "heldout_rollouts":0,
        "geometry_review_status":geometry.status,
        "geometry_review_sha256":geometry.source_sha256,
        "scientific_freeze_permitted":False,
        "heldout_evaluation_permitted":False,
        "screens":screens,
    }
    raw=(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n").encode("utf-8")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as f:
        f.write(raw)
    print(json.dumps({"output":str(path),"sha256":sha256(raw).hexdigest(),
        "run_count":2,"heldout_rollouts":0,"scientific_freeze_permitted":False,
        "all_screens_pass":all(x["all_train_track_screens_pass"] for x in screens)},indent=2),flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
