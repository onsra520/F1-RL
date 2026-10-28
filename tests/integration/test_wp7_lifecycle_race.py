"""Versioned lifecycle race on software fixtures; never TRAIN policy evaluation."""
from dataclasses import replace

import numpy as np
import pytest

from src.rl.wp7_race import WP7RaceConfig
from src.simulation.ballistic_flight import FlightState
from tests.integration.test_pv5_lifecycle import lifecycle_config
from tests.integration.test_wp7_race_contract import race_factory


def lifecycle_race_factory(*, launch=True, finish=2.5, max_steps=10):
    base = race_factory(checkpoints=(.2,.7),finish=finish,speed=5.,max_steps=max_steps)
    initial = replace(base.initial_state,flight=FlightState("takeoff" if launch else "grounded"))
    if launch:
        initial = replace(initial,vehicle=replace(initial.vehicle,
            velocity=replace(initial.vehicle.velocity,y=2.)),
            suspension=replace(initial.suspension,heave_velocity_mps=2.))
    return replace(base,initial_state=initial,physics_config=lifecycle_config(),
                   race_config=WP7RaceConfig(version="wp7_directed_lifecycle_research_v1"))


def test_takeoff_airborne_gates_landing_and_finish_use_one_gym_episode():
    env = lifecycle_race_factory().create()
    try:
        env.reset()
        events = []
        crossed = []
        times = []
        for tick in range(10):
            _,_,terminated,truncated,info = env.step(np.array([0.,0.],dtype=np.float32))
            events.extend(e['kind'] for e in info['physics']['transition']['events'])
            crossed.extend(info['race']['directed_crossings'])
            times.extend(info['race']['crossing_time_origin_s']+t
                         for t in info['race']['tick_local_crossing_times_s'])
            if terminated or truncated:
                break
        assert terminated and not truncated
        assert events == ["takeoff","landing"]
        assert crossed == ["cp-0","cp-1","finish"]
        assert times[:2] == pytest.approx([.04,.14])
        assert info['telemetry']['grounded']
    finally:
        env.close()


def test_airborne_finish_precedes_same_tick_timeout():
    base = race_factory(checkpoints=(.1,.2),finish=.3,speed=5.,max_steps=1)
    candidate = lifecycle_race_factory(finish=.9,max_steps=1)
    candidate = replace(candidate,track=base.track,navigation=base.navigation)
    env = candidate.create()
    try:
        env.reset()
        _,_,terminated,truncated,info = env.step(np.array([0.,0.],dtype=np.float32))
        assert terminated and not truncated
        assert not info['telemetry']['grounded']
        assert info['race']['tick_local_crossing_times_s'] == pytest.approx((.02,.04,.06))
    finally:
        env.close()


def lifecycle_reset_factory():
    from tests.integration.test_wp7_train_reset import randomized_factory
    base = randomized_factory()
    return replace(base,physics_config=lifecycle_config(),
        initial_state=replace(base.initial_state,flight=FlightState("grounded")),
        race_config=WP7RaceConfig(version="wp7_directed_lifecycle_research_v1"),
        reset_config=replace(base.reset_config,version="wp7_train_reset_lifecycle_research_v1"))


@pytest.mark.parametrize("backend",["dummy","subproc"])
def test_lifecycle_reset_replays_and_retains_vector_rank_streams(backend):
    from src.training.env_factory import TrainingScenario, TrainingScenarioSet
    from src.training.vector_env import make_multi_track_vector_env
    from src.training.config import VectorEnvConfig
    candidate = lifecycle_reset_factory()
    scenarios = TrainingScenarioSet((TrainingScenario(candidate.reset_track_id,candidate),))
    vec = make_multi_track_vector_env(scenarios,VectorEnvConfig(n_envs=2,backend=backend),23)
    try:
        previous = None
        for index in range(2):
            vec.seed(99)
            vec.reset()
            info = vec.step(np.zeros((2,2),dtype=np.float32))[-1]
            draws = [r['train_reset'] for r in info]
            assert [d['rank'] for d in draws] == [0,1]
            assert all(d['reset_index'] == index for d in draws)
            assert draws[0]['draw_seed'] != draws[1]['draw_seed']
            assert draws[0]['draw_seed'] != previous
            previous = draws[0]['draw_seed']
            assert all(r['physics']['transition']['final_phase']=='grounded' for r in info)
    finally:
        vec.close()


def test_actual_gym_crossing_uses_ballistic_height_at_thin_aperture():
    from src.simulation.geometry.vector import Vec3
    candidate = lifecycle_race_factory()
    cp = candidate.track.checkpoints[0]
    cp = replace(cp,volume=replace(cp.volume,center=Vec3(0.,.5877375,.25),
                                  half_extents=Vec3(5.,1e-6,.001)))
    candidate = replace(candidate,track=replace(candidate.track,
        checkpoints=(cp,candidate.track.checkpoints[1])),
        navigation=replace(candidate.navigation,points=(candidate.navigation.points[0],cp.volume.center,
            *candidate.navigation.points[2:])))
    env = candidate.create()
    try:
        env.reset()
        info = env.step(np.array([0.,0.],dtype=np.float32))[-1]
        assert info['race']['completed_checkpoint_ids'] == ('cp-0',)
        assert info['race']['tick_local_crossing_times_s'] == pytest.approx((.05,))
    finally:
        env.close()


def test_old_race_and_reset_versions_cannot_silently_enable_lifecycle():
    candidate = lifecycle_reset_factory()
    with pytest.raises(ValueError,match="grounded"):
        replace(candidate,race_config=WP7RaceConfig())
    with pytest.raises(ValueError,match="reset"):
        replace(candidate,reset_config=replace(candidate.reset_config,version="wp7_train_reset_research_v1"))
    with pytest.raises(ValueError,match="unregistered"):
        WP7RaceConfig(version="wp7_directed_lifecycle_research_v1",hard_violation="terminate_failure")


def test_lifecycle_reset_reproduces_without_restarting_from_gym_seed():
    a,b = lifecycle_reset_factory().create(),lifecycle_reset_factory().create()
    try:
        prior = None
        for index in range(2):
            first,ia = a.reset(seed=3)
            second,ib = b.reset(seed=999)
            assert np.array_equal(first,second)
            assert ia['train_reset'] == ib['train_reset']
            assert ia['train_reset']['reset_index'] == index
            assert ia['train_reset']['draw_seed'] != prior
            prior = ia['train_reset']['draw_seed']
    finally:
        a.close()
        b.close()


def test_missing_timed_physics_path_cannot_publish_or_earn_finish():
    from src.core.physics import PhysicsTransitionError
    from src.simulation.world import step_world
    from src.simulation.vehicle import VehicleControl
    candidate = lifecycle_race_factory()
    config = candidate.physics_config
    result = step_world(candidate.track,candidate.initial_state,VehicleControl(0.,0.,0.),
        candidate.vehicle_parameters,.1,1.,**config.world_options())
    with pytest.raises(PhysicsTransitionError,match="trajectory"):
        config.validate_transition(replace(result,center_segments=()),previous=candidate.initial_state,
            parameters=candidate.vehicle_parameters,dt=.1,start_time_s=0.,end_time_s=.1)


def test_new_lifecycle_registration_still_blocks_all_training(tmp_path):
    from src.training.runner import TrainingRunner
    from src.training.config import TrainingRunConfig, VectorEnvConfig
    from src.rl.algorithms.ppo import PPOConfig
    from src.validation.wp7_admission import ScientificAdmissionError
    candidate = lifecycle_reset_factory()
    config = TrainingRunConfig(PPOConfig(),VectorEnvConfig(n_envs=1),artifacts_root=tmp_path/'forbidden')
    with pytest.raises(ScientificAdmissionError):
        TrainingRunner(config).run_single(candidate)
    assert not config.artifacts_root.exists()


def test_lifecycle_hard_route_violation_remains_invalid_pending_owner_rule():
    from src.rl.wp7_race import RaceContractError
    from tests.integration.test_wp7_race_contract import sideways_factory
    base = sideways_factory(.5)
    candidate = replace(base,physics_config=lifecycle_config(),
        initial_state=replace(base.initial_state,flight=FlightState('grounded')),
        race_config=WP7RaceConfig(version='wp7_directed_lifecycle_research_v1'))
    env = candidate.create()
    try:
        env.reset()
        assert env.step(np.array([0.,0.],dtype=np.float32))[-1]['race']['soft_zone_encountered']
        with pytest.raises(RaceContractError,match='hard_violation'):
            env.step(np.array([0.,0.],dtype=np.float32))
        assert env.last_research_outcome['kind'] == 'uncertified_route_state'
        assert env.last_research_outcome['valid_sample'] is False
    finally:
        env.close()


def test_automatic_edge_release_orders_airborne_gate_in_same_tick():
    base = race_factory(checkpoints=(3.6,),finish=10.,speed=40.)
    candidate = replace(base,physics_config=lifecycle_config(),
        track=replace(base.track,surfaces=(replace(base.track.surfaces[0],length=4.),)),
        initial_state=replace(base.initial_state,flight=FlightState('grounded')),
        race_config=WP7RaceConfig(version='wp7_directed_lifecycle_research_v1'))
    env = candidate.create()
    try:
        env.reset()
        info = env.step(np.array([0.,0.],dtype=np.float32))[-1]
        departure = info['physics']['partial_support']['takeoff_at_s']
        assert info['race']['completed_checkpoint_ids'] == ('cp-0',)
        assert info['race']['tick_local_crossing_times_s'][0] > departure
        assert info['race']['tick_local_crossing_times_s'] == pytest.approx((.09,))
        assert not info['telemetry']['grounded']
    finally:
        env.close()


def test_rebound_preserves_gate_order_and_time_after_impact():
    from src.core.simulation import SimulationConfig
    base = race_factory(checkpoints=(.25,1.1),finish=1.2,speed=5.,max_steps=1)
    initial = replace(base.initial_state,flight=FlightState('airborne'),
        vehicle=replace(base.initial_state.vehicle,position=replace(base.initial_state.vehicle.position,y=.85),
                        velocity=replace(base.initial_state.vehicle.velocity,y=-1.),grounded=False),
        suspension=replace(base.initial_state.suspension,height_m=.85))
    config = lifecycle_config()
    config = replace(config,ballistic_flight_policy=replace(config.ballistic_flight_policy,
        manifold=replace(config.ballistic_flight_policy.manifold,restitution=.5)))
    candidate = replace(base,initial_state=initial,physics_config=config,
        simulation_config=SimulationConfig(.25,1.),
        navigation=replace(base.navigation,points=(initial.vehicle.position,*base.navigation.points[1:])),
        race_config=WP7RaceConfig(version='wp7_directed_lifecycle_research_v1'))
    env = candidate.create()
    try:
        env.reset()
        _,_,terminated,truncated,info = env.step(np.array([0.,0.],dtype=np.float32))
        assert terminated and not truncated
        assert [e['kind'] for e in info['physics']['transition']['events']] == ['rebound']
        assert info['race']['directed_crossings'] == ('cp-0','cp-1','finish')
        times = info['race']['tick_local_crossing_times_s']
        assert times == pytest.approx((.05,.22,.24))
        impact = info['physics']['transition']['events'][0]['time_s']
        assert times[0] < impact < times[1]
        assert not info['telemetry']['grounded']
    finally:
        env.close()
