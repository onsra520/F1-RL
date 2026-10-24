"""Development owner race truncates checkpoint/reward at applied crash TOI."""
from dataclasses import replace
import pytest
from f1rl.sandbox.accepted_contact_transition import apply_certified_face_impulse
from f1rl.sandbox.physical_crash import first_applied_grounded_crash
from f1rl.sandbox.rigid_body_impact_response import RigidBodyImpactMass
from f1rl.sandbox.body_collision import VehicleBodyGeometry
from f1rl.sandbox.center_trajectory import CenterSegment
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.track.volume import OrientedBoxVolume
from f1rl.sandbox.entities.obstacle import Obstacle
from f1rl.runtime.simulation import SimulationSnapshot
from f1rl.runtime.physics import PhysicsTransitionAudit
from f1rl.learning.route_navigation import AuthoredRoute
from f1rl.learning.episode import RaceProgressState,RaceRoute
from f1rl.learning.wp7_race import WP7RaceConfig,advance_directed_race,RaceContractError
from tests.integration.test_wp7_race_contract import race_factory
Q=Quaternion.identity()
Z=Vec3(0.,0.,0.)
BODY=VehicleBodyGeometry(.1,.2,.1,Z)
MASS=RigidBodyImpactMass(1000.,Vec3(500.,500.,500.))

def model():
    fac=race_factory(checkpoints=(.2,.6),finish=.8,speed=5.)
    obs=Obstacle("crash-1",OrientedBoxVolume(Vec3(0.,.5,.4),Vec3(2.,2.,.1),Q))
    a=replace(fac.initial_state.vehicle,position=Vec3(0.,.5,0.),velocity=Vec3(0.,0.,5.))
    b=replace(a,position=Vec3(0.,.5,.5))
    original=(CenterSegment(0.,.1,a.position,b.position),)
    collision=first_applied_grounded_crash((a,b),original,(obs,),BODY,MASS,fixed_dt_s=.1)
    assert collision is not None
    world=replace(fac.initial_state,vehicle=collision.contact.postimpact)
    snap=SimulationSnapshot(1,.1,world,fac.track.query_point(world.vehicle.position,1.,world.track),(),
         physics_transition=PhysicsTransitionAudit('wp7_transition_audit_v1','wp7_lifecycle_v1',
               'grounded','grounded',0.,.1,('grounded_face_impulse_absorbing_v4_development',),()),
         center_segments=collision.prefix,physical_crash=collision)
    return fac,snap,collision

def test_race_and_checkpoint_list_do_not_cross_after_obstacle_crash():
    fac,snap,hit=model()
    result=advance_directed_race(
      fac.track,RaceRoute.from_track(fac.track),AuthoredRoute(fac.navigation),
      RaceProgressState(0,False),snap,
      WP7RaceConfig(version='wp7_directed_lifecycle_crash_dev_v3',
                    hard_violation='terminate_valid_failure'))
    assert result.terminal_failure is True
    assert result.terminal_reason=="physical_crash"
    assert result.terminal_time_s==pytest.approx(hit.tick_contact_time_s)
    assert result.completed_checkpoint_ids==("cp-0",)
    assert result.directed_crossings==("cp-0",)
    assert result.progress.state.finished is False
    assert all(t<=hit.tick_contact_time_s for t in result.crossing_times_s)

def test_old_v2_owner_race_fails_closed_for_new_crash_evidence():
    fac,snap,hit=model()
    with pytest.raises(RaceContractError,match="distinct versioned"):
        advance_directed_race(
          fac.track,RaceRoute.from_track(fac.track),AuthoredRoute(fac.navigation),
          RaceProgressState(0,False),snap,
          WP7RaceConfig(version='wp7_directed_lifecycle_owner_v2',
                        hard_violation='terminate_valid_failure'))
