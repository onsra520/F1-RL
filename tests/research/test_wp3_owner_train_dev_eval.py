"""Owner-approved TRAIN-development Eval must be source/seed bound and fail closed."""
from __future__ import annotations
import json
from pathlib import Path
import pytest

from f1rl.research import wp3_owner_train_dev_eval as od
from f1rl.research.wp3_eval_only_admission import MODELS, TRACKS, STRATA, digest


def make_roster():
    rows = []
    for m in MODELS:
        for t in TRACKS:
            for s in STRATA:
                rows.append({
                    "model_id":m, "track_id":t, "stratum":s,
                    "candidate_id":m[0],"seed_index":int(m[-1]),
                    "episodes":[{"episode":i,
                                 "environment_seed":1000000+TRACKS.index(t)*200+
                                   STRATA.index(s)*60+i,
                                 "policy_seed":2000000+TRACKS.index(t)*200+
                                   STRATA.index(s)*60+i} for i in range(60)]})
    roster={
        "evaluation":{
            "tracks":list(TRACKS),"strata":list(STRATA),"hard_route_boundary_m":4.0,
            "episodes_per_track_stratum_model":60,
            "checkpoint_selection":"final_checkpoint_only",
            "policy_actions":"stochastic_with_paired_candidate_independent_rng_seeds",
            "heldout_access":False},
        "expected_cells":96,"expected_episodes":5760,"seed_rows":rows}
    roster["sha256"]=digest(roster)
    return roster


def fixture(monkeypatch,tmp_path):
    root=tmp_path/"source"
    hist=tmp_path/"original"
    root.mkdir()
    hist.mkdir()
    (hist/"evaluation").mkdir()
    roster=make_roster()
    (hist/"evaluation/evaluation-roster.json").write_text(json.dumps(roster))
    models=[]
    for mid in MODELS:
        p=hist/"runs"/mid/"model-final.zip"
        p.parent.mkdir(parents=True)
        p.write_bytes(mid.encode())
        models.append({"model_id":mid,"model_final_sha256":od.hash_file(p)})
    prov={"models":models,"training_source_commit":od.TRAIN_HEAD,
          "evaluation_roster_sha256":roster["sha256"]}
    (hist/"WP3-EVAL-SOURCE-PROVENANCE.json").write_text(json.dumps(prov))
    monkeypatch.setattr(od,"_source",lambda root:"frozen-source")
    monkeypatch.setattr(od,"preflight",lambda root,history: (
        {"current_evaluation_source_commit":"frozen-source",
         "frozen_final_checkpoint_count":12},
        roster,prov))
    return root,hist


def test_exact_96_cells_and_5760_slots_paired_seeds():
    c=od.plan_roster(make_roster())
    assert len(c)==96
    assert sum(len(row["episodes"]) for row in c)==5760
    assert [x["model_id"] for x in c[:8]]==["A-seed0"]*8
    assert c[0]["track"]=="straight-sprint-v1"
    assert c[1]["stratum"]=="randomized_train"
    assert c[8]["model_id"]=="A-seed1"
    assert c[-1]["model_id"]=="D-seed2"
    for t in TRACKS:
        for s in STRATA:
            assert len({tuple((x["environment_seed"],x["policy_seed"]) for x in cell["episodes"])
                        for cell in c if cell["track"]==t and cell["stratum"]==s})==1


def test_tamper_roster_rejected():
    r=make_roster()
    r["seed_rows"][0]["episodes"][1]["policy_seed"]+=1
    with pytest.raises(ValueError, match="SHA"):
        od.plan_roster(r)


def test_log_parser_detects_duplicate_reorder_and_seed_forgery():
    cell=od.plan_roster(make_roster())[0]
    a=od._event("r",cell,0,"episode_started",
                environment_seed=cell["episodes"][0]["environment_seed"],
                policy_seed=cell["episodes"][0]["policy_seed"])
    b=od._event("r",cell,0,"episode_completed",
                result={"status":"ENGINEERING_EPISODE_COMPLETED","outcome":"finish"})
    rows,pending=od.parse_cell([a,b],cell,"r")
    assert len(rows)==1 and pending is None
    with pytest.raises(ValueError):
        od.parse_cell([a,a,b],cell,"r")
    with pytest.raises(ValueError):
        od.parse_cell([b],cell,"r")
    with pytest.raises(ValueError):
        od.parse_cell([a,b,b],cell,"r")
    with pytest.raises(ValueError,match="paired seed"):
        od.parse_cell([dict(a,policy_seed=12),b],cell,"r")
    with pytest.raises(ValueError):
        od.parse_cell([dict(a,episode=1)],cell,"r")
    with pytest.raises(ValueError):
        od.parse_cell([dict(a,run_id="foreign"),b],cell,"r")


def test_started_without_result_stays_pending():
    cell=od.plan_roster(make_roster())[0]
    a=od._event("r",cell,0,"episode_started",
                environment_seed=cell["episodes"][0]["environment_seed"],
                policy_seed=cell["episodes"][0]["policy_seed"])
    results,pending=od.parse_cell([a],cell,"r")
    assert results==[] and pending==a


@pytest.mark.parametrize("id,approval",[
    ("WP3-OFFICIAL-20261024-001","I approve the internal TRAIN-development evaluation"),
    ("WP3-OWNER-TRAINDEV-test-run","fake independent reviewer approved"),
    ("WP3-OWNER-TRAINDEV-../traversal","I approve the internal TRAIN-development evaluation"),
])
def test_invalid_or_forged_owner_registration_is_rejected(monkeypatch,tmp_path,id,approval):
    root,hist=fixture(monkeypatch,tmp_path)
    with pytest.raises(ValueError):
        od.new_run(root,hist,tmp_path/"out",id,approval)
    assert not (tmp_path/"out").exists()


def test_new_run_persisted_immutable_inputs_and_non_overwrite(monkeypatch,tmp_path):
    root,hist=fixture(monkeypatch,tmp_path)
    out=tmp_path/"out"
    doc=od.new_run(root,hist,out,"WP3-OWNER-TRAINDEV-unit-test","I approve the internal TRAIN-development evaluation")
    assert doc["expected_episode_slots"]==5760
    assert doc["scientific_evidence"] is False
    assert doc["scientific_registered_evaluation"] is False
    assert doc["independent_physics_review"] is None
    assert len(doc["cells"])==96
    assert (out/"RUN-MANIFEST.json").is_file()
    assert od.load_run(root,hist,out)["sha256"]==doc["sha256"]
    assert od.snapshot(root,hist,out)["unattempted_slots"]==5760
    with pytest.raises(FileExistsError):
        od.new_run(root,hist,out,"WP3-OWNER-TRAINDEV-unit-test","I approve the internal TRAIN-development evaluation")


def test_original_model_digest_tamper_rejected(monkeypatch,tmp_path):
    root,hist=fixture(monkeypatch,tmp_path)
    m=hist/"runs"/"B-seed2"/"model-final.zip"
    m.write_bytes(b"wrong")
    with pytest.raises(ValueError,match="mutated"):
        od.new_run(root,hist,tmp_path/"out","WP3-OWNER-TRAINDEV-test-digest",
                   "I approve the internal TRAIN-development evaluation")


def test_run_manifest_tamper_fails_load(monkeypatch,tmp_path):
    root,hist=fixture(monkeypatch,tmp_path)
    out=tmp_path/"out"
    od.new_run(root,hist,out,"WP3-OWNER-TRAINDEV-test-tamper",
               "I approve the internal TRAIN-development evaluation")
    p=out/"RUN-MANIFEST.json"
    d=json.loads(p.read_text())
    d["scientific_evidence"]=True
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="checksum"):
        od.load_run(root,hist,out)


def test_invalid_episode_does_not_count_as_valid_completion(monkeypatch,tmp_path):
    root,hist=fixture(monkeypatch,tmp_path)
    out=tmp_path/"out"
    doc=od.new_run(root,hist,out,"WP3-OWNER-TRAINDEV-test-invalid",
               "I approve the internal TRAIN-development evaluation")
    c=doc["cells"][0]
    ev=out/"cells"/od.cell_filename(c)
    seed=c["episodes"][0]
    od._append(ev,od._event(doc["run_id"],c,0,"episode_started",
               environment_seed=seed["environment_seed"],policy_seed=seed["policy_seed"]))
    od._append(ev,od._event(doc["run_id"],c,0,"episode_invalid",
               result={"status":"ENGINEERING_EPISODE_INVALID",
                       "outcome":"invalid","error":"physics"}))
    s=od.snapshot(root,hist,out)
    assert s["valid_episodes"]==0
    assert s["invalid_episodes"]==1
    assert s["unattempted_slots"]==5759
    assert s["complete_cells"]==0
    assert not s["all_cells_complete"]


def test_workers_bounded():
    assert od.MAX_WORKERS<=3


def test_cli_compiles():
    path=Path(__file__).resolve().parents[2]/"scripts/wp3_owner_train_dev_eval.py"
    compile(path.read_text("utf8"),str(path),"exec")


def test_preserve_exact_frozen_roster_cell_order():
    roster=make_roster()
    roster["seed_rows"].sort(key=lambda x: (x["model_id"] != "C-seed0", x["model_id"], x["track_id"], x["stratum"]))
    roster["sha256"]=digest({k:v for k,v in roster.items() if k != "sha256"})
    cells=od.plan_roster(roster)
    assert cells[0]["model_id"]=="C-seed0"
    assert [(x["model_id"],x["track"],x["stratum"]) for x in cells] == [
      (x["model_id"],x["track_id"],x["stratum"]) for x in roster["seed_rows"]]


def test_incomplete_report_is_not_scientific_ranking(monkeypatch,tmp_path):
    from f1rl.research import wp3_owner_train_dev_reports as reports
    root,hist=fixture(monkeypatch,tmp_path)
    out=tmp_path/"output"
    manifest=od.new_run(root,hist,out,"WP3-OWNER-TRAINDEV-test-reports",
              "I approve the internal TRAIN-development evaluation")
    before=od.snapshot(root,hist,out)
    monkeypatch.setattr(reports,"_learning_progress_svg",lambda *args:"<svg></svg>\n")
    prov=json.loads((hist/"WP3-EVAL-SOURCE-PROVENANCE.json").read_text())
    for m in prov["models"]:m["training_telemetry_sha256"]="fixture-no-training"
    doc=reports.write_reports(out,hist,manifest,before,prov)
    assert doc["status"]=="INCOMPLETE_RANKING_WITHHELD"
    assert all(v["status"]=="WITHHELD_INCOMPLETE_96_CELL_EVAL" for v in doc["candidate_descriptive_only"].values())
    assert "WITHHELD" in (out/"WP3-CANDIDATE-COMPARISON.md").read_text()
    assert (out/"WP3-OUTPUT-SHA256-MANIFEST.json").is_file()
    assert (out/"WP3-MODEL-COMPARISON.svg").is_file()
    with pytest.raises(FileExistsError):
        reports.write_reports(out,hist,manifest,before,prov)


def test_probe_cell_append_only_and_idempotent_no_redraw(monkeypatch,tmp_path):
    root,hist=fixture(monkeypatch,tmp_path)
    out=tmp_path/"scratch"
    m=od.new_run(root,hist,out,"WP3-OWNER-TRAINDEV-probe-only",
                 "I approve the internal TRAIN-development evaluation")
    cell=m["cells"][0]
    got=od.execute_cell(str(root),str(hist),str(out),m["run_id"],cell,probe=True)
    assert got["status"]=="CELL_COMPLETE" and got["valid"]==60
    filename=out/"cells"/od.cell_filename(cell)
    first=filename.read_bytes()
    assert len(od.read_events(filename))==120
    second=od.execute_cell(str(root),str(hist),str(out),m["run_id"],cell,probe=True)
    assert second["status"]=="CELL_COMPLETE"
    assert filename.read_bytes()==first
    assert od.snapshot(root,hist,out)["valid_episodes"]==60


def test_dangling_started_is_invalidated_without_rerun(monkeypatch,tmp_path):
    root,hist=fixture(monkeypatch,tmp_path)
    out=tmp_path/"scratch"
    m=od.new_run(root,hist,out,"WP3-OWNER-TRAINDEV-interrupt-case",
                 "I approve the internal TRAIN-development evaluation")
    c=m["cells"][0]
    ev=out/"cells"/od.cell_filename(c)
    s=c["episodes"][0]
    od._append(ev,od._event(m["run_id"],c,0,"episode_started",
        environment_seed=s["environment_seed"],policy_seed=s["policy_seed"]))
    got=od.execute_cell(str(root),str(hist),str(out),m["run_id"],c,probe=True)
    assert got["status"]=="ABORTED_INVALID" and got["valid"]==0
    events=od.read_events(ev)
    assert [e["event"] for e in events]==["episode_started","episode_invalid"]
    assert events[-1]["result"]["error_type"]=="InterruptedAttempt"
    assert od.snapshot(root,hist,out)["invalid_episodes"]==1
