"""Scientific WP7 recovery: TRAIN-only candidate factory builder contract."""
from pathlib import Path
import copy
import yaml
import numpy as np
import pytest

from src.rl.reconstruction import RecoveryCandidate, build_training_scenarios
from src.rl.offtrack import OffTrackConfig
from src.rl.rewards import RouteProgressReward, OffTrackPenalty

CANDIDATE=Path("configs/recovery/wp7-c2-candidate.yml")
TRAIN=("straight-sprint-v1","slalom-v1","right-turn-v1","ramp-v1")
HELDOUT=("left-turn-v1","offset-slalom-v1")


def test_candidate_contract_references_unfrozen_legacy_environment() -> None:
    candidate=RecoveryCandidate.from_yaml(CANDIDATE)
    assert candidate.status == "candidate"
    assert candidate.experiment.metadata.status == "draft"
    assert candidate.experiment.training.protocol_status == "pilot_owned"
    assert candidate.behavior["navigation"] == "authored-surface-polyline-endpoint-v1"
    assert candidate.behavior["reward"] == "route-highwater-progress-collision-time-offtrack-v1"


def test_builder_instantiates_only_4_training_factories_with_identical_contract() -> None:
    candidate=RecoveryCandidate.from_yaml(CANDIDATE)
    scenarios=build_training_scenarios(candidate)
    assert tuple(s.track_id for s in scenarios.scenarios)==TRAIN
    assert not (set(HELDOUT) & {s.track_id for s in scenarios.scenarios})
    assert scenarios.action_space.shape==(2,)
    assert scenarios.observation_space.shape==(16,)
    for entry in scenarios.scenarios:
        factory=entry.factory
        assert factory.navigation is not None
        assert factory.offtrack_config==OffTrackConfig(6.0,10)
        assert tuple(type(term) for term in factory.reward_composer.terms)==(
            RouteProgressReward,
            __import__("src.rl.rewards",fromlist=["CollisionPenalty"]).CollisionPenalty,
            __import__("src.rl.rewards",fromlist=["TimePenalty"]).TimePenalty,
            OffTrackPenalty,
        )
        env=factory.create()
        try:
            obs,info=env.reset(seed=17)
            assert obs.shape==(16,)
            assert env.observation_space.contains(obs)
            next_obs,reward,done,truncated,step=env.step(np.array([0.0,0.0],dtype=np.float32))
            assert next_obs.shape==(16,)
            assert set(step["reward_components"])=={"route_progress","collision","time","offtrack"}
            assert step["offtrack_consecutive_steps"]==0
            assert np.isfinite(reward)
        finally:
            env.close()


@pytest.mark.parametrize("edit",[
    lambda d: d.update(status="frozen"),
    lambda d: d.update(unrecognized=True),
    lambda d: d["behavior"].update(navigation="direct-checkpoint"),
    lambda d: d["rewards"].update(route_progress_weight_per_meter=-1),
    lambda d: d["offtrack"].update(max_consecutive_steps=0),
    lambda d: d["offtrack"].update(lateral_limit_meters=float("nan")),
])
def test_candidate_loader_fails_closed_on_invalid_protocol(tmp_path: Path,edit) -> None:
    document=yaml.safe_load(CANDIDATE.read_text(encoding="utf-8"))
    document["base_experiment"]=str((Path.cwd()/"configs/experiments/ppo-generalization-v1.yml").resolve())
    edit(document)
    changed=tmp_path/"candidate.yml"
    changed.write_text(yaml.safe_dump(document),encoding="utf-8")
    with pytest.raises(ValueError):
        RecoveryCandidate.from_yaml(changed)


def test_candidate_does_not_redefine_final_seeds_or_reclassify_tracks() -> None:
    candidate=RecoveryCandidate.from_yaml(CANDIDATE)
    assert candidate.experiment.seeds.ppo==()
    assert candidate.experiment.seeds.sampling==()
    assert candidate.experiment.tracks.heldout_ids==HELDOUT
    assert candidate.experiment.tracks.training_ids==TRAIN
