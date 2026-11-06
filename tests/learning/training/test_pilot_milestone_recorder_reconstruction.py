"""200k TRAIN-only checkpoint/final evidence collection using MOCK models.

No PPO model is trained and no heldout policy is evaluated by these tests.
"""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

import f1rl.learning.training.runner as runner_module
from f1rl.learning.pilot_milestones import PilotMilestoneRecorder
from f1rl.learning.pilot_preflight import TRAIN_IDS
from f1rl.learning.reconstruction import RecoveryCandidate, build_training_scenarios
from f1rl.learning.training.config import TrainingRunConfig
from f1rl.learning.training.runner import TrainingRunner

from tests.learning.training.mocks.mock_ppo import MockPPOModel
from tests.learning.training.mocks.mock_vec_env import MockVecEnv

ROOT=Path.cwd()
CANDIDATE=ROOT/"configs/recovery/wp7-c2-candidate.yml"
PROFILE=ROOT/"configs/recovery/wp7-pilot-200k.yml"


def _runner(tmp_path,monkeypatch,evaluator,progress=None):
    MockPPOModel.reset()
    MockPPOModel.progress_points=(
        tuple(range(25_000,200_001,25_000)) if progress is None else progress
    )
    monkeypatch.setattr(runner_module,"PPOModel",MockPPOModel)
    monkeypatch.setattr(runner_module,"make_multi_track_vector_env",lambda *args:MockVecEnv())
    recorder=PilotMilestoneRecorder(
        repository_root=ROOT,
        candidate_path=CANDIDATE,
        runtime_config_path=PROFILE,
        mock_mode=True,
        evaluator=evaluator,
    )
    config=replace(TrainingRunConfig.from_yaml(PROFILE),artifacts_root=tmp_path/"runs")
    runner=TrainingRunner(
        config,checkpoint_observer=recorder.on_checkpoint,
        final_model_observer=recorder.on_final,
    )
    scenarios=build_training_scenarios(RecoveryCandidate.from_yaml(CANDIDATE))
    return runner,scenarios,recorder


def _mock_eval(*,repository_root,candidate_path,runtime_config_path,checkpoint_path,episodes_per_track,test_policy=None):
    archive=Path(checkpoint_path)
    return {
        "schema_version":"wp7-recovery-train-checkpoint-v1",
        "checkpoint_name":archive.name,
        "checkpoint_sha256":sha256(archive.read_bytes()).hexdigest(),
        "policy_source":"injected_test_policy_not_scientific_evidence",
        "scientific_evidence":False,
        "track_ids":list(TRAIN_IDS),
        "episode_count_per_track":episodes_per_track,
        "track_summaries":[{"track_id":i, "success_count":0} for i in TRAIN_IDS],
        "all_train_tracks_successful":False,
        "heldout_rollouts":0,
        "candidate_source_fingerprints":list(PilotMilestoneRecorder(
            repository_root=ROOT,candidate_path=CANDIDATE,
            runtime_config_path=PROFILE,mock_mode=True,evaluator=lambda **kw:{}
        ).source_fingerprints),
    }


def test_complete_200k_mock_collects_8_verified_reports_and_manifest(tmp_path,monkeypatch):
    runner,scenarios,recorder=_runner(tmp_path,monkeypatch,_mock_eval)
    result=runner.run_multi(scenarios)
    reports=sorted((result.run_directory/"evaluations").glob("milestone-*.json"))
    assert len(reports)==8
    assert len(result.checkpoint_paths)==7
    assert recorder.milestones_recorded==(25_000,50_000,75_000,100_000,125_000,150_000,175_000,200_000)
    assert result.actual_total_timesteps==200_000
    manifest=json.loads((result.run_directory/"evaluations"/"manifest.json").read_text("utf-8"))
    assert manifest["status"]=="mock_verified_not_scientific_evidence"
    assert manifest["milestones"]==list(recorder.milestones_recorded)
    assert manifest["heldout_rollouts"]==0
    assert len(manifest["reports"])==8
    for record in manifest["reports"]:
        report_path=result.run_directory/record["path"]
        assert report_path.exists()
        assert sha256(report_path.read_bytes()).hexdigest()==record["sha256"]
        report=json.loads(report_path.read_text("utf-8"))
        assert report["source_fingerprints"]==list(recorder.source_fingerprints)
        assert report["checkpoint_sha256"]==sha256((result.run_directory/report["model_path"]).read_bytes()).hexdigest()
        assert report["scientific_evidence"] is False


def test_failed_checkpoint_evaluator_leaves_failed_run_without_false_report(tmp_path,monkeypatch):
    def evaluator(**kwargs):
        if kwargs["checkpoint_path"].name.endswith("step50000.zip"):
            raise RuntimeError("mock evaluation failed")
        return _mock_eval(**kwargs)
    runner,scenarios,_=_runner(tmp_path,monkeypatch,evaluator)
    with pytest.raises(RuntimeError,match="mock evaluation failed"):
        runner.run_multi(scenarios)
    run=next((tmp_path/"runs").iterdir())
    assert json.loads((run/"state.json").read_text("utf-8"))["status"]=="failed"
    assert len(list((run/"evaluations").glob("milestone-*.json")))==1
    assert not (run/"evaluations"/"manifest.json").exists()


def test_tampered_archive_hash_rejected_before_publishing(tmp_path,monkeypatch):
    def evaluator(**kwargs):
        report=_mock_eval(**kwargs)
        kwargs["checkpoint_path"].write_bytes(b"CHANGED_AFTER_EVALUATION")
        return report
    runner,scenarios,_=_runner(tmp_path,monkeypatch,evaluator)
    with pytest.raises(ValueError,match="sha256"):
        runner.run_multi(scenarios)
    run=next((tmp_path/"runs").iterdir())
    assert json.loads((run/"state.json").read_text("utf-8"))["status"]=="failed"
    assert not list((run/"evaluations").glob("milestone-*.json"))


def test_missing_200k_final_milestone_rejected(tmp_path,monkeypatch):
    runner,scenarios,_=_runner(tmp_path,monkeypatch,_mock_eval,progress=tuple(range(25_000,200_000,25_000)))
    with pytest.raises(ValueError,match="final milestone"):
        runner.run_multi(scenarios)
    run=next((tmp_path/"runs").iterdir())
    assert json.loads((run/"state.json").read_text("utf-8"))["status"]=="failed"
    assert not (run/"evaluations"/"manifest.json").exists()


def test_mock_mode_must_be_explicit_with_open_preflight_blockers():
    with pytest.raises(RuntimeError,match="review"):
        PilotMilestoneRecorder(
            repository_root=ROOT,candidate_path=CANDIDATE,
            runtime_config_path=PROFILE,evaluator=_mock_eval,
        )


def test_modified_earlier_checkpoint_rejected_before_next_report(tmp_path,monkeypatch):
    runner,scenarios,recorder=_runner(tmp_path,monkeypatch,_mock_eval)
    ordinary=recorder.on_checkpoint
    def tamper_before_second(archive,steps):
        if steps==50_000:
            previous=archive.parent/(archive.name.replace("step50000","step25000"))
            assert previous.is_file()
            previous.write_bytes(b"tampered after evidence publication")
        ordinary(archive,steps)
    runner._checkpoint_observer=tamper_before_second
    with pytest.raises(ValueError,match="checkpoint sha256"):
        runner.run_multi(scenarios)
    run=next((tmp_path/"runs").iterdir())
    assert json.loads((run/"state.json").read_text("utf-8"))["status"]=="failed"
    assert len(list((run/"evaluations").glob("milestone-*.json")))==1


def test_modified_prior_report_rejected_before_next_report(tmp_path,monkeypatch):
    runner,scenarios,recorder=_runner(tmp_path,monkeypatch,_mock_eval)
    ordinary=recorder.on_checkpoint
    def tamper_before_second(archive,steps):
        if steps==50_000:
            report=archive.parent.parent/"evaluations"/"milestone-025000.json"
            assert report.is_file()
            report.write_text('{"tampered": true}',encoding="utf-8")
        ordinary(archive,steps)
    runner._checkpoint_observer=tamper_before_second
    with pytest.raises(ValueError,match="report sha256"):
        runner.run_multi(scenarios)


def test_mock_final_actual_timesteps_can_overshoot_200k_by_rollout(tmp_path,monkeypatch):
    steps=tuple(range(25_000,200_000,25_000))+(200_704,)
    runner,scenarios,recorder=_runner(tmp_path,monkeypatch,_mock_eval,progress=steps)
    result=runner.run_multi(scenarios)
    assert result.actual_total_timesteps==200_704
    final=json.loads((result.run_directory/"evaluations"/"milestone-200000.json").read_text("utf-8"))
    assert final["milestone_timesteps"]==200_000
    assert final["actual_timesteps"]==200_704
    assert final["scientific_evidence"] is False


def test_finished_mock_run_can_be_verified_after_copy_and_detects_tampering(tmp_path,monkeypatch):
    from f1rl.learning.pilot_milestones import verify_pilot_evidence_manifest
    runner,scenarios,_=_runner(tmp_path,monkeypatch,_mock_eval)
    result=runner.run_multi(scenarios)
    summary=verify_pilot_evidence_manifest(result.run_directory)
    assert summary["verified_milestones"]==8
    assert summary["status"]=="verified_mock_evidence_not_scientific"
    assert summary["heldout_rollouts"]==0

    early_model=result.checkpoint_paths[0]
    original=early_model.read_bytes()
    early_model.write_bytes(b"later corruption")
    with pytest.raises(ValueError,match="checkpoint sha256"):
        verify_pilot_evidence_manifest(result.run_directory)
    early_model.write_bytes(original)
    report=result.run_directory/"evaluations"/"milestone-025000.json"
    report.write_text("{}",encoding="utf-8")
    with pytest.raises(ValueError,match="report sha256"):
        verify_pilot_evidence_manifest(result.run_directory)


def test_exploratory_real_mode_uses_explicit_unreviewed_semantics(monkeypatch):
    from f1rl.learning import pilot_milestones as module
    monkeypatch.setattr(module,"_git_revision",lambda root:("8"*40,False))
    recorder=module.PilotMilestoneRecorder(
        repository_root=ROOT,candidate_path=CANDIDATE,
        runtime_config_path=PROFILE,exploratory_mode=True,
    )
    assert recorder._mock is False
    assert recorder._exploratory is True
    assert recorder._preflight["review_blockers"]
    assert recorder._evaluator is module.evaluate_training_checkpoint
    with pytest.raises(ValueError,match="real checkpoint evaluator"):
        module.PilotMilestoneRecorder(
            repository_root=ROOT,candidate_path=CANDIDATE,
            runtime_config_path=PROFILE,exploratory_mode=True,
            evaluator=_mock_eval,
        )
    with pytest.raises(ValueError,match="mutually exclusive"):
        module.PilotMilestoneRecorder(
            repository_root=ROOT,candidate_path=CANDIDATE,
            runtime_config_path=PROFILE,exploratory_mode=True,
            mock_mode=True,evaluator=_mock_eval,
        )
