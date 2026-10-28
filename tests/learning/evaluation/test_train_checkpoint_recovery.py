"""C2 pilot evaluation: mock-test only TRAIN track boundary and archive fingerprints."""
from hashlib import sha256
from pathlib import Path

import pytest

from src.evaluation.metrics import EpisodeMetrics
from src.rl.reconstruction import RecoveryCandidate
from src.rl import train_checkpoint as module

ROOT=Path.cwd()
CANDIDATE=ROOT/"configs/recovery/wp7-c2-candidate.yml"
PROFILE=ROOT/"configs/recovery/wp7-pilot-200k.yml"
TRAIN=("straight-sprint-v1","slalom-v1","right-turn-v1","ramp-v1")


class DummyPolicy:
    def predict(self,observation,*,deterministic=True):
        raise AssertionError("mock-only evaluator must not drive simulation")


def _metrics(track_id:str) -> EpisodeMetrics:
    return EpisodeMetrics(
        track_id=track_id, success=False, truncated=True,
        steps=1, episode_time=0.1, completion_time=None,
        episode_return=0.0, reward_component_totals=(),
        collision_count=0,path_length=0.0,
        final_normalized_route_progress=0.0,
        mean_speed=0.0,max_speed=0.0,
        airborne_steps=0, boost_applied_steps=0,slow_zone_applied_steps=0
    )


def _evaluate(monkeypatch,archive,episodes=1):
    seen=[]
    def simulate(*,track_id,factory,policy):
        seen.append(track_id)
        assert policy is not None
        assert factory.navigation is not None
        return _metrics(track_id)
    monkeypatch.setattr(module,"evaluate_episode",simulate)
    report=module.evaluate_training_checkpoint(
        repository_root=ROOT,
        candidate_path=CANDIDATE,
        runtime_config_path=PROFILE,
        checkpoint_path=archive,
        episodes_per_track=episodes,
        test_policy=DummyPolicy(),
    )
    return report,seen


def test_mock_checkpoint_evaluator_fingerprints_and_evaluates_exact_train_tracks(monkeypatch,tmp_path) -> None:
    archive=tmp_path/"mock-model.zip"
    archive.write_bytes(b"FAKE_ARCHIVE_ONLY_NOT_A_REAL_MODEL")
    result,seen=_evaluate(monkeypatch,archive,episodes=2)
    assert seen==[track for track in TRAIN for _ in range(2)]
    assert result["track_ids"]==list(TRAIN)
    assert result["checkpoint_sha256"]==sha256(archive.read_bytes()).hexdigest()
    assert result["episode_count_per_track"]==2
    assert len(result["track_summaries"])==4
    assert result["policy_source"]=="injected_test_policy_not_scientific_evidence"
    assert result["scientific_evidence"] is False
    assert result["all_train_tracks_successful"] is False


def test_missing_checkpoint_rejected_before_evaluation(monkeypatch,tmp_path) -> None:
    def no_eval(*args,**kwargs):
        raise AssertionError("should not reach simulation")
    monkeypatch.setattr(module,"evaluate_episode",no_eval)
    with pytest.raises(ValueError,match="checkpoint"):
        module.evaluate_training_checkpoint(
            repository_root=ROOT,candidate_path=CANDIDATE,
            runtime_config_path=PROFILE,
            checkpoint_path=tmp_path/"missing.zip",
            test_policy=DummyPolicy()
        )


@pytest.mark.parametrize("count",[0,-1,True,1.5])
def test_episode_count_rejected(count,tmp_path) -> None:
    archive=tmp_path/"valid.zip"
    archive.write_bytes(b"fake")
    with pytest.raises(ValueError,match="episodes"):
        module.evaluate_training_checkpoint(
            repository_root=ROOT,candidate_path=CANDIDATE,
            runtime_config_path=PROFILE,
            checkpoint_path=archive,
            episodes_per_track=count,
            test_policy=DummyPolicy(),
        )


def test_incorrect_track_id_is_rejected(monkeypatch,tmp_path) -> None:
    archive=tmp_path/"valid.zip"
    archive.write_bytes(b"fake")
    monkeypatch.setattr(module,"evaluate_episode",lambda **kwargs:_metrics("left-turn-v1"))
    with pytest.raises(ValueError,match="track"):
        module.evaluate_training_checkpoint(
            repository_root=ROOT,candidate_path=CANDIDATE,
            runtime_config_path=PROFILE,
            checkpoint_path=archive,test_policy=DummyPolicy()
        )
