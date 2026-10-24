"""Actual opt-in Simulation -> Gym -> owner-race -> evaluator applied crash."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest

from f1rl.learning.evaluation import evaluate_episode
from f1rl.learning.evaluation.evaluator import _step_evidence,_valid_terminal_failure
from f1rl.learning.evaluation.crash_contract import validate_crash_terminal_info
from f1rl.learning.rewards import RewardComposer,TimePenalty
from f1rl.learning.wp7_race import WP7RaceConfig
from f1rl.runtime.physics import PhysicsTransitionError
from f1rl.runtime.simulation import Simulation
from f1rl.sandbox.ballistic_flight import FlightState
from f1rl.sandbox.entities.obstacle import Obstacle
from f1rl.sandbox.track.volume import OrientedBoxVolume
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.spatial_contact_dynamics import _suspension_state
from f1rl.sandbox.vehicle import VehicleControl
from tests.integration.test_spatial_ground_motion import spatial_factory
from tests.learning.evaluation.mocks.policies import ConstantPolicy


def crash_factory():
    f=spatial_factory()
    physics=replace(f.physics_config,mode="wp7_lifecycle_v1",
        ground_motion_version="spatial_road_edge_partial_lifecycle_v4",
        independent_suspension_policy=replace(
            f.physics_config.independent_suspension_policy,
            force_model="road_relative_unilateral_v3"))
    v=replace(f.initial_state.vehicle,velocity=Vec3(0.,0.,5.))
    state=replace(f.initial_state,vehicle=v,suspension=_suspension_state(v),
                  flight=FlightState("grounded"))
    obstacle=Obstacle("crash-1",OrientedBoxVolume(
        Vec3(0.,.8,6.4),Vec3(2.,2.,.1),v.orientation))
    return replace(f,physics_config=physics,initial_state=state,
        track=replace(f.track,obstacles=(obstacle,)),
        race_config=WP7RaceConfig(
            version="wp7_directed_lifecycle_crash_dev_v3",
            hard_violation="terminate_valid_failure"),
        reward_composer=RewardComposer((TimePenalty(2.),)))


def test_real_simulator_state_contains_applied_impulse_audited_and_absorbing():
    f=crash_factory()
    sim=Simulation(f.track,f.initial_state,f.vehicle_parameters,
                   f.simulation_config,physics_config=f.physics_config)
    snap=sim.step(VehicleControl(0.,0.,0.))
    impact=snap.physical_crash
    assert impact is not None
    assert snap.world.vehicle==impact.contact.postimpact
    assert snap.world.vehicle.velocity.z<impact.contact.preimpact.velocity.z
    assert snap.physics_transition.solvers[-1]=="grounded_face_impulse_absorbing_v4_development"
    assert impact.contact.energy_lost_j>0.
    assert snap.center_segments[-1].start==snap.center_segments[-1].end
    assert snap.center_segments[-1].end_time_s==pytest.approx(.1)


def test_actual_evaluator_counts_crash_as_development_failure_and_only_prefix_reward():
    f=crash_factory()
    steps=[]
    result=evaluate_episode(track_id="test-applied-crash",
        factory=f,policy=ConstantPolicy(np.zeros(3,dtype=np.float32)),
        step_observer=steps.append)
    assert result.success is False and result.truncated is False
    assert result.steps==1 and result.collision_count==1
    assert result.episode_return==pytest.approx(-.121467590332,abs=2e-5)
    assert len(steps)==1 and steps[0]["terminated"] is True
    info=steps[0]["info"]
    assert _valid_terminal_failure(info)
    assert validate_crash_terminal_info(info)
    assert info["research_outcome"]["reason"]=="physical_crash_development"
    assert info["physical_crash_terminal"]["scientific_evidence"] is False
    assert info["race"]["reward_valid_prefix_s"]==pytest.approx(
        info["physical_crash_terminal"]["impact_time_s"])
    assert info["telemetry"]["obstacle_hit_ids"]==("crash-1",)
    assert info["race"]["directed_crossings"]==()


@pytest.mark.parametrize(("section","key","value"),[
    ("physical_crash_terminal","impact_time_s",.099),
    ("physical_crash_terminal","energy_lost_j",999999.),
    ("physical_crash_terminal","linear_impulse_n_s",(float("nan"),0.,0.)),
    ("physical_crash_terminal","accepted_velocity_mps",(0.,0.,99.)),
    ("physical_crash_terminal","scientific_evidence",True),
    ("physical_crash_terminal","version","faked"),
    ("race","reward_valid_prefix_s",.1),
    ("race","terminal_reason","finish"),
    ("race","tick_local_crossing_times_s",(.095,)),
    ("physics","sha256","none"),
])
def test_changed_crash_physics_and_prefix_never_mint_score(section,key,value):
    f=crash_factory()
    env=f.create()
    try:
        env.reset()
        info=env.step(np.zeros(3,dtype=np.float32))[-1]
    finally:env.close()
    forged=deepcopy(info)
    forged[section][key]=value
    assert not validate_crash_terminal_info(forged)
    assert not _valid_terminal_failure(forged)
    with pytest.raises(ValueError,match="unregistered physical crash terminal"):
        _step_evidence(forged,executed_steps=1,fixed_dt=.1)


def test_original_v2_owner_still_fails_closed_at_synthetic_physical_crash():
    f=crash_factory()
    old=replace(f,race_config=WP7RaceConfig(
        version="wp7_directed_lifecycle_owner_v2",
        hard_violation="terminate_valid_failure"))
    env=old.create()
    try:
        env.reset()
        with pytest.raises(Exception,match="distinct versioned"):
            env.step(np.zeros(3,dtype=np.float32))
    finally:env.close()


def test_real_simulation_is_absorbing_after_accepted_crash():
    f=crash_factory()
    sim=Simulation(f.track,f.initial_state,f.vehicle_parameters,
                   f.simulation_config,physics_config=f.physics_config)
    res=sim.step(VehicleControl(0.,0.,0.))
    assert res.physical_crash is not None
    with pytest.raises(RuntimeError,match="absorbing physical crash"):
        sim.step(VehicleControl(0.,0.,0.))
    assert sim.reset().physical_crash is None


@pytest.mark.parametrize("forgery",["post_velocity","impulse","energy","event_time","prefix","unscored_event"])
def test_physics_transition_independently_rejects_tampered_applied_impact(forgery):
    from f1rl.sandbox.world import step_world
    from f1rl.sandbox.geometry.vector import Vec3
    f=crash_factory()
    result=step_world(f.track,f.initial_state,VehicleControl(0.,0.,0.),
         f.vehicle_parameters,f.simulation_config.fixed_dt,
         f.simulation_config.max_surface_distance,
         **f.physics_config.world_options())
    assert result.crash is not None
    certified=result.crash
    outcome=certified.contact
    if forgery=="post_velocity":
        result=replace(result,state=replace(result.state,
            vehicle=replace(result.state.vehicle,velocity=Vec3(0.,0.,999.))))
    elif forgery=="impulse":
        result=replace(result,crash=replace(certified,contact=replace(
            outcome,impulse_n_s=Vec3(0.,0.,1.))))
    elif forgery=="energy":
        result=replace(result,crash=replace(certified,contact=replace(
            outcome,energy_lost_j=99999.)))
    elif forgery=="event_time":
        result=replace(result,crash=replace(certified,tick_contact_time_s=.099))
    elif forgery=="prefix":
        result=replace(result,center_segments=result.precrash_result.center_segments)
    elif forgery=="unscored_event":
        from f1rl.sandbox.interactions.events import ObstacleHit
        result=replace(result,events=(ObstacleHit('crash-1'),))
    with pytest.raises(PhysicsTransitionError):
        f.physics_config.validate_transition(result,
            previous=f.initial_state,parameters=f.vehicle_parameters,
            dt=f.simulation_config.fixed_dt,start_time_s=0.,
            end_time_s=f.simulation_config.fixed_dt)
