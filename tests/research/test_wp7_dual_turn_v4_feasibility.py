"""Reproducible v4 engineering demo preflight, without PPO or HELD-OUT."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from f1rl.learning.pilot_preflight import TRAIN_IDS
from f1rl.sandbox.geometry.vector import Vec3

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def replay_runner():
    path=ROOT/"f1rl/cli/evidence/wp7_dual_turn_v4_feasibility.py"
    spec=importlib.util.spec_from_file_location("wp7_dual_turn_v4_feasibility_demo",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("track_id,sign",[
    ("right-turn-train-v2",1),
    ("left-turn-train-v2",-1),
])
def test_two_v2_engineering_factories_run_actual_v4_physics_with_intact_race(replay_runner,track_id,sign):
    from f1rl.research.wp3_eval_only_coordinator import FULL06_RUNTIME_PROFILE
    factory,binding=replay_runner.build_factory(track_id)
    assert binding["requested_profile"]==FULL06_RUNTIME_PROFILE
    assert binding["effective"]=={
        "physics_version":"spatial_road_edge_partial_lifecycle_v4",
        "race_version":"wp7_directed_lifecycle_crash_contract_v4",
        "terminal_contract":"wp3_physical_crash_terminal_v1",
        "max_abs_tilt_rad":.7,
    }
    assert factory.requires_scientific_admission
    assert factory.reset_config is None
    assert factory.reset_track_id is None
    assert factory.initial_state.vehicle.position == Vec3(-3*sign,.5,5)
    pieces,finish_s=replay_runner.control_path(track_id)
    assert len(pieces)==3
    assert finish_s>140

    env=factory.create()
    try:
        observation,info=env.reset(seed=20261106)
        assert env.observation_space.contains(observation)
        assert info["physics"]["configuration"]["ground_motion_version"]=="spatial_road_edge_partial_lifecycle_v4"
        assert info["race"]["configuration"]["version"]=="wp7_directed_lifecycle_crash_contract_v4"
        action=np.asarray([0,.5,0],dtype=np.float32)
        next_obs,_,terminated,truncated,info=env.step(action)
        assert env.observation_space.contains(next_obs)
        assert not terminated and not truncated
        assert info["physics"]["configuration"]["ground_motion_version"]=="spatial_road_edge_partial_lifecycle_v4"
        assert info["race"]["configuration"]["version"]=="wp7_directed_lifecycle_crash_contract_v4"
        assert info["telemetry"]["grounded"]
    finally:
        env.close()


def test_historical_four_train_registry_not_replaced_by_unregistered_dual_v2_demo():
    assert TRAIN_IDS==(
        "straight-sprint-v1","slalom-v1","right-turn-v1","ramp-v1"
    )
    assert not (ROOT/"assets/tracks/manifest.json").read_text().count("right-turn-train-v2")
