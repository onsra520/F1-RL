"""Actual accepted rigid state updates, momentum/energy and timed prefix."""
from dataclasses import replace
import pytest
from f1rl.sandbox.accepted_contact_transition import (
    apply_certified_face_impulse,apply_certified_bump_stop_impulse,
    absorb_crash_center_prefix
)
from f1rl.sandbox.body_collision import VehicleBodyGeometry
from f1rl.sandbox.center_trajectory import CenterSegment
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.rigid_body_impact_response import RigidBodyImpactMass
from f1rl.sandbox.track.patch import PlanarSurfacePatch
from f1rl.sandbox.track.volume import OrientedBoxVolume
from f1rl.sandbox.vehicle import VehicleState
Q=Quaternion.identity()
Z=Vec3(0.,0.,0.)
MASS=RigidBodyImpactMass(1000.,Vec3(500.,500.,500.))
HULL=VehicleBodyGeometry(.4,.4,.4,Z)
OBSTACLE=OrientedBoxVolume(Z,Vec3(.1,2.,2.),Q)
ROAD=PlanarSurfacePatch("flat",Z,Vec3(0.,0.,1.),Vec3(0.,1.,0.),10.,10.)

def pose(x,y,vx,vy):
    return VehicleState(Vec3(x,y,0.),Vec3(vx,vy,0.),Q,Z,True)


def test_obstacle_impulse_applied_to_new_vehicle_state_without_changing_pose():
    first,last=pose(.6,0.,-2.,0.),pose(0.,0.,-2.,0.)
    event=apply_certified_face_impulse(first,last,HULL,OBSTACLE,MASS,duration_s=.3)
    assert event.kind=="obstacle_face"
    assert event.first_clear_s<event.witnessed_contact_s
    assert event.postimpact.position==event.preimpact.position
    assert -2.<event.postimpact.velocity.x<0.  # extremal hull corner also spins
    assert event.preimpact.velocity.x==pytest.approx(-2.)
    assert event.postimpact.angular_velocity.magnitude>0.
    assert 0.<event.energy_lost_j<2000.
    assert event.kinetic_after_j<event.kinetic_before_j


def test_bump_stop_impulse_applied_to_new_vehicle_state_and_kinetic_drops():
    first,last=pose(0.,-.11,0.,-.2),pose(0.,-.13,0.,-.2)
    event=apply_certified_bump_stop_impulse(
        first,last,MASS,wheel_local=Z,road=ROAD,max_bump_m=.12,duration_s=.1)
    assert event.kind=="wheel_hard_stop"
    assert event.postimpact.position==event.preimpact.position
    assert event.postimpact.velocity.y==pytest.approx(0.,abs=1e-8)
    assert event.energy_lost_j==pytest.approx(20.)
    assert event.first_clear_s<event.witnessed_contact_s


def test_impact_replaces_future_motion_with_absorbing_fixed_position():
    first,last=pose(.6,0.,-2.,0.),pose(0.,0.,-2.,0.)
    event=apply_certified_face_impulse(first,last,HULL,OBSTACLE,MASS,duration_s=.3)
    segments=(CenterSegment(0.,.3,first.position,last.position),)
    prefix=absorb_crash_center_prefix(segments,event.witnessed_contact_s,
                                     fixed_dt_s=.3,accepted=event)
    assert len(prefix)==2
    assert prefix[0].end_time_s==pytest.approx(event.witnessed_contact_s)
    assert prefix[1].start==prefix[1].end==event.postimpact.position
    assert prefix[-1].end_time_s==.3


def test_prefix_omits_post_contact_checkpoint_without_inventing_an_earlier_one():
    first,last=pose(.6,0.,-2.,0.),pose(0.,0.,-2.,0.)
    event=apply_certified_face_impulse(first,last,HULL,OBSTACLE,MASS,duration_s=.3)
    segments=(CenterSegment(0.,.3,first.position,last.position),)
    shortened=absorb_crash_center_prefix(segments,event.witnessed_contact_s,
                                        fixed_dt_s=.3,accepted=event)
    far_gate=OrientedBoxVolume(Vec3(.05,0.,0.),Vec3(.01,2.,2.),Q)
    assert all(not s.directed_crossings(far_gate) for s in shortened)


def test_mismatched_prefix_or_unsupported_contact_cannot_be_accepted():
    first,last=pose(.6,0.,-2.,0.),pose(0.,0.,-2.,0.)
    event=apply_certified_face_impulse(first,last,HULL,OBSTACLE,MASS,duration_s=.3)
    with pytest.raises(ValueError,match="contact pose"):
        absorb_crash_center_prefix((CenterSegment(0.,.3,Vec3(10.,0.,0.),
                                      Vec3(11.,0.,0.)),),
                                   event.witnessed_contact_s,fixed_dt_s=.3,accepted=event)
