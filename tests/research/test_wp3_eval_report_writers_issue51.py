"""Synthetic report-only regression; never touches campaign evidence or real evaluation."""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from src.interfaces.evaluation import _summarize, _write_evaluation_reports


def test_all_report_chart_writers_complete_and_are_idempotent():
    evaluation={
        "tracks": ["straight-sprint-v1","slalom-v1","right-turn-v1","ramp-v1"],
        "strata":["nominal","randomized_train"],
    }
    models=[]
    cells=[]
    with TemporaryDirectory(prefix="wp3-issue51-report-smoke-") as td:
        root=Path(td)
        for c in "ABCD":
            for seed in range(3):
                model_id=f"{c}-seed{seed}"
                models.append({"model_id":model_id,"candidate_id":c,"seed_index":seed})
                telemetry=root/"runs"/model_id/"training-telemetry.jsonl"
                telemetry.parent.mkdir(parents=True)
                telemetry.write_text("".join(json.dumps({"event_type":"ppo_update","metrics":{"train/loss":float(k)/10}})+"\n" for k in range(1,4)),encoding="utf-8")
                for track in evaluation["tracks"]:
                    for stratum in evaluation["strata"]:
                        cell={"model_id":model_id,"candidate_id":c,"seed_index":seed,
                              "track_id":track,"stratum":stratum,"status":"completed",
                              "success_count":60,
                              "mean_route_progress":1.0,"mean_completed_checkpoints":2,
                              "collision_rate":0.0,"mean_collision_count":0.0,
                              "hard_route_violation_rate":0.0,"timeout_rate":0.0,
                              "mean_steering_rate_rms_rad_per_s":1.0,
                              "mean_episode_duration_seconds":20.0,
                              "failure_reasons":{"valid_finish":60}}
                        cells.append(cell)
                        p=root/"evaluation"/"cells"/model_id/f"{track}-{stratum}.jsonl"
                        p.parent.mkdir(parents=True,exist_ok=True)
                        p.write_text(json.dumps({"schema_version":"f1rl_wp3_train_eval_event_v1","event":"episode_completed","record":{"valid":True,"metrics":{"success":True}}})+"\n",encoding="utf-8")
        assert len(cells)==96
        result={"status":"PASS","comparison":_summarize(models,cells,evaluation),
                "invalid_cells":[],"total_valid_episode_rows":5760,"completed_cells":96}
        _write_evaluation_reports(root,result,models,cells,evaluation)
        expected=("WP3-CANDIDATE-COMPARISON.md",
                  "WP3-STATISTICAL-READINESS.md","WP3-FINAL-STATUS.md",
                  "WP3-MODEL-COMPARISON.svg",
                  "WP3-FAILURE-DISTRIBUTION.svg","WP3-LEARNING-PROGRESS.svg")
        first={k:(root/k).read_bytes() for k in expected}
        assert all(first.values())
        assert {name: sha256(content).hexdigest() for name, content in first.items()} == {
            "WP3-CANDIDATE-COMPARISON.md": "90adedbb4c18479ced8aab8039b86b828140e0b383ce4a0b352c4c163b2fb8fb",
            "WP3-STATISTICAL-READINESS.md": "941dda7266d11cc87fa8059625fef58e5b46a240bfb0175929bb5b1d6f3ea527",
            "WP3-FINAL-STATUS.md": "a3fa4e062d0d977d6cbe743cab69172d5d58a66341212b38cc4c4de326a2fe37",
            "WP3-MODEL-COMPARISON.svg": "3b157202db256b99e1ea9b3292dbd8a6c698cc6bed7c21cba6bd356e39e5e71d",
            "WP3-FAILURE-DISTRIBUTION.svg": "bc777aea69b29a4f54a47a0bc805c2d505b7ef042d921735637a04f0b968e878",
            "WP3-LEARNING-PROGRESS.svg": "1e2868778cb2483de0f650c53ecb33b929f1dd104ebc6f2a78d93bf9407cf233",
        }
        _write_evaluation_reports(root,result,models,cells,evaluation)
        assert all((root/k).read_bytes()==first[k] for k in expected)
        assert len(expected) == 6
