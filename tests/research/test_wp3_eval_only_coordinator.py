"""Adversarial safety tests for the independent, fail-closed Eval-only coordinator."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import pytest

from src.research import wp3_eval_only_coordinator as co
from src.research.wp3_eval_only_admission import MODELS, STRATA, TRACKS


def roster():
    return {"seed_rows": [
        {"model_id": model, "track_id": track, "stratum": stratum,
         "seed_index": int(model[-1]),
         "episodes": [
             {"episode": i, "environment_seed": 100000 + 400 * TRACKS.index(track)
               + 60 * STRATA.index(stratum) + i,
              "policy_seed": 200000 + 400 * TRACKS.index(track)
               + 60 * STRATA.index(stratum) + i}
             for i in range(60)]}
        for model in MODELS for track in TRACKS for stratum in STRATA]}


def fake_fixture(monkeypatch, tmp_path):
    root, hist = tmp_path / "source", tmp_path / "historical"
    root.mkdir()
    hist.mkdir()
    (hist / "evaluation").mkdir()
    r = roster()
    r["sha256"] = "original-roster"
    (hist / "evaluation" / "evaluation-roster.json").write_text(json.dumps(r))
    models = []
    for model in MODELS:
        path = hist / "runs" / model / "model-final.zip"
        path.parent.mkdir(parents=True)
        path.write_bytes(("PPO-"+model).encode())
        models.append({"model_id": model, "model_final_sha256": co.hash_file(path)})
    p = {"training_source_commit": co.TRAIN_HEAD, "models": models}
    (hist / "WP3-EVAL-SOURCE-PROVENANCE.json").write_text(json.dumps(p))
    monkeypatch.setattr(co, "preflight", lambda *args: (
        {"source_dirty": False, "frozen_final_checkpoint_count": 12,
         "current_evaluation_source_commit": "clean-test-head", "sha256": "preflight-sha"},
        r, p))
    monkeypatch.setattr(co, "refuse_full_eval",
                        lambda check: (_ for _ in ()).throw(RuntimeError("FULL_EVAL_BLOCKED: missing independent certification")))
    monkeypatch.setattr(co, "_source", lambda *args: "clean-test-head")
    return root, hist


def test_deterministic_original_seed_schedule():
    cases = co.roster_cases(roster(), episode_index=4)
    assert len(cases) == 16
    assert len(set((x["model_id"], x["track"], x["stratum"], x["episode"]) for x in cases)) == 16
    assert all(x["episode"] == 4 for x in cases)
    assert cases[0]["track"] == "straight-sprint-v1"


@pytest.mark.parametrize("model_ids,index", [
    (("X-seed0",), 4), (("A-seed1","A-seed1"), 4),
    (("A-seed1",), -1), (("A-seed1",), 60),
    (("A-seed1",), True), ((), 4)])
def test_bad_model_or_index_fails_closed(model_ids,index):
    with pytest.raises(ValueError):
        co.roster_cases(roster(), model_ids=model_ids, episode_index=index)


def test_missing_roster_cell_fails_closed():
    original = roster()
    original["seed_rows"].pop()
    with pytest.raises(ValueError):
        co.roster_cases(original)


def test_forged_paired_seed_is_rejected():
    original = roster()
    cell = next(r for r in original["seed_rows"] if r["model_id"] == "C-seed0"
                and r["track_id"] == "slalom-v1" and r["stratum"] == "nominal")
    cell["episodes"][4]["policy_seed"] += 1
    with pytest.raises(ValueError, match="paired seed"):
        co.roster_cases(original)


def test_engineering_run_writes_append_only_evidence(monkeypatch,tmp_path):
    root, hist = fake_fixture(monkeypatch,tmp_path)
    out = tmp_path / "new"
    def evaluate(root,history,case,cache):
        return {"success": True, "failure_reason": None,
                "final_normalized_route_progress": 1.,
                "episode_time": 20., "completed_checkpoints": 2,
                "collision_count": 0, "timeout": False}
    doc = co.run_engineering(root,hist,out,"WP3-ENG-test-v1",evaluate=evaluate)
    assert doc["planned"] == 16 and doc["completed"] == 16 and doc["invalid"] == 0
    assert len(list((out/"cells").glob("*.jsonl"))) == 16
    assert doc["full_5760_eval_gate"] == "BLOCKED"
    assert doc["official_eval_started"] is False
    assert doc["scientific_evidence"] is False
    assert (out/"episodes.csv").is_file()
    assert (out/"REPORT.md").is_file()
    assert co._json(out/"WP3-ENGINEERING-EVALUATION.json")["sha256"] == co.digest({
        k:v for k,v in doc.items() if k != "sha256"})
    with pytest.raises(FileExistsError):
        co.run_engineering(root,hist,out,"WP3-ENG-test-v1",evaluate=evaluate)
    ev = json.loads(next((out/"cells").glob("*.jsonl")).read_text().splitlines()[0])
    assert ev["event"] == "episode_started"


def test_bad_episode_preserved_as_invalid_no_retry(monkeypatch,tmp_path):
    root, hist = fake_fixture(monkeypatch,tmp_path)
    calls=[]
    def evaluate(root,history,case,cache):
        calls.append(case)
        if len(calls) == 2:raise ValueError("real physics exception")
        return {"success": False, "failure_reason": "physical_crash",
                "final_normalized_route_progress": .5, "episode_time": 8.}
    doc=co.run_engineering(root,hist,tmp_path/"out","WP3-ENG-with-invalid",evaluate=evaluate)
    assert len(calls)==16 and doc["invalid"]==1
    assert doc["completed"]==15
    cell=sorted((tmp_path/"out"/"cells").glob("*.jsonl"))[1]
    events=[json.loads(x) for x in cell.read_text().splitlines()]
    assert [x["event"] for x in events]==["episode_started","episode_invalid"]
    assert events[1]["record"]["error_type"]=="ValueError"
    assert doc["full_5760_eval_gate"]=="BLOCKED"


def test_frozen_model_hash_mismatch_fails_before_output(monkeypatch,tmp_path):
    root,hist=fake_fixture(monkeypatch,tmp_path)
    p=hist/"runs"/"A-seed1"/"model-final.zip"
    p.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint changed"):
        co.run_engineering(root,hist,tmp_path/"new","WP3-ENG-tamper",evaluate=lambda *a: {})
    assert not (tmp_path/"new").exists()


def test_run_id_restrictions(monkeypatch,tmp_path):
    root,hist=fake_fixture(monkeypatch,tmp_path)
    with pytest.raises(ValueError):
        co.run_engineering(root,hist,tmp_path/"bad","WP3-REGISTERED-5760")
    with pytest.raises(ValueError):
        co.run_engineering(root,hist,tmp_path/"bad","WP3-ENG-../../evil")
    assert not (tmp_path/"bad").exists()


def test_full_eval_hard_gate_propagates(monkeypatch,tmp_path):
    root,hist=fake_fixture(monkeypatch,tmp_path)
    with pytest.raises(RuntimeError,match="FULL_EVAL_BLOCKED"):
        co.forbid_registered_full_eval(root,hist)


def test_cli_entrypoint_parses_as_python():
    script = Path(__file__).resolve().parents[2] / "scripts/wp3_eval_only_coordinator.py"
    compile(script.read_text(encoding="utf8"), str(script), "exec")


def test_real_admission_gate_checksum_survives_bound_source_head(monkeypatch, tmp_path):
    from src.research.wp3_eval_only_admission import refuse_full_eval as original_gate
    root, history = tmp_path / "root", tmp_path / "history"
    root.mkdir()
    history.mkdir()
    monkeypatch.setattr(co, "_source", lambda *args: "new-eval-commit")
    monkeypatch.setattr(co, "build_eval_only_preflight", lambda *args: {
        "source_dirty": False, "frozen_final_checkpoint_count": 12,
        "sha256": "original_preflight_hash", "mandatory_replay_gate": "BLOCKED",
    })
    monkeypatch.setattr(co, "_json", lambda path: {
        "training_source_commit": co.TRAIN_HEAD,
    })
    monkeypatch.setattr(co, "_roster_check", lambda *args: None)
    bound, _, _ = co.preflight(root, history)
    assert bound["current_evaluation_source_commit"] == "new-eval-commit"
    assert bound["sha256"] == co.digest({k: v for k, v in bound.items() if k != "sha256"})
    with pytest.raises(RuntimeError, match="FULL_EVAL_BLOCKED"):
        original_gate(bound)
