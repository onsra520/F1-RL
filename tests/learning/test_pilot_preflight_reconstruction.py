"""Recovery pilot preparation: immutable fingerprints, seeds and risk gates, no training."""
from hashlib import sha256
from pathlib import Path

import pytest

from src.learning.pilot_preflight import build_pilot_preflight

ROOT=Path.cwd()
CANDIDATE=ROOT/"configs/recovery/wp7-c2-candidate.yml"
PROFILE=ROOT/"configs/recovery/wp7-pilot-200k.yml"
TRACKS=("straight-sprint-v1","slalom-v1","right-turn-v1","ramp-v1")


def _preflight(profile=PROFILE):
    return build_pilot_preflight(
        repository_root=ROOT, candidate_path=CANDIDATE,
        runtime_config_path=profile
    )


def test_recovery_preflight_has_200k_train_only_protocol_and_digests() -> None:
    report=_preflight()
    assert report["status"]=="review_required"
    assert report["training_track_ids"]==list(TRACKS)
    assert report["training_seed_pairs"]==[
        {"ppo":0,"sampling":0,"role":"calibration"},
        {"ppo":1,"sampling":1,"role":"confirmation"},
    ]
    assert report["requested_total_timesteps"]==200000
    assert report["checkpoint_every_steps"]==25000
    assert report["evaluation_milestones"]==list(range(25000,200001,25000))
    assert report["vector_env"]=={"n_envs":8,"backend":"dummy","start_method":"spawn"}
    assert report["ppo_device"]=="cpu"
    assert report["final_seed_replicates_executed"] is False
    assert report["trained_models"]==0
    assert report["heldout_rollouts"]==0
    assert len(report["fingerprints"])==8
    for item in report["fingerprints"]:
        path=ROOT/item["path"]
        assert path.is_file()
        assert item["sha256"]==sha256(path.read_bytes()).hexdigest()


def test_preflight_warns_on_corners_and_slalom_clearance() -> None:
    report=_preflight()
    by_track={item["track_id"]:item for item in report["route_audit"]}
    assert set(by_track)==set(TRACKS)
    assert by_track["right-turn-v1"]["maximum_heading_change_degrees"]==pytest.approx(90)
    assert 1.8 < by_track["slalom-v1"]["minimum_obstacle_clearance_meters"] < 2.0
    assert len(report["review_blockers"])>=2
    assert any("right-turn-v1" in issue for issue in report["review_blockers"])
    assert any("slalom-v1" in issue for issue in report["review_blockers"])


def test_preflight_deterministic_without_mutating_artifacts() -> None:
    first=_preflight()
    second=_preflight()
    # Git dirtiness and timestamps are not permitted inside deterministic reports.
    assert first==second
    assert not (ROOT/"artifacts/runs"/"preflight-train").exists()


@pytest.mark.parametrize("change", [
    lambda data: data["training"].update(total_timesteps=50000),
    lambda data: data["training"].update(checkpoint_every_steps=20000),
    lambda data: data["ppo"].update(device="cuda"),
    lambda data: data["vector_env"].update(n_envs=4),
    lambda data: data["training"].update(sampling_seed=200),
    lambda data: data["ppo"].update(seed=100),
])
def test_preflight_rejects_drifted_runtime_profile(tmp_path:Path,change) -> None:
    import yaml
    document=yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    change(document)
    profile=tmp_path/"wrong.yml"
    profile.write_text(yaml.safe_dump(document),encoding="utf-8")
    with pytest.raises(ValueError,match="pilot"):
        _preflight(profile)
