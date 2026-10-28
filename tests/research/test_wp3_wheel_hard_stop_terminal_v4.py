"""Development-only applied wheel hard-stop absorbing crash cannot be forged."""
from dataclasses import replace
import pytest

from src.sandbox.accepted_contact_transition import (
    apply_certified_bump_stop_impulse,absorb_crash_center_prefix,
)
from src.sandbox.center_trajectory import CenterSegment
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.geometry.vector import Vec3
from src.sandbox.rigid_body_impact_response import RigidBodyImpactMass
from src.sandbox.track.patch import PlanarSurfacePatch
from src.sandbox.vehicle import VehicleState
from src.sandbox.physical_crash import PhysicalCrash
from src.learning.crash_terminal_registered import SOURCES

Q=Quaternion.identity()
Z=Vec3(0.,0.,0.)
M=RigidBodyImpactMass(1000.,Vec3(500.,500.,500.))
ROAD=PlanarSurfacePatch("stop",Z,Vec3(0.,0.,1.),Vec3(0.,1.,0.),20.,20.)

def pose(y):
    return VehicleState(Vec3(0.,y,0.),Vec3(0.,-.2,0.),Q,Z,True)


def event():
    a,b=pose(-.11),pose(-.13)
    accepted=apply_certified_bump_stop_impulse(
        a,b,M,wheel_local=Z,road=ROAD,max_bump_m=.12,duration_s=.1)
    segments=(CenterSegment(0.,.1,a.position,b.position),)
    prefix=absorb_crash_center_prefix(
        segments,accepted.witnessed_contact_s,
        fixed_dt_s=.1,accepted=accepted)
    crash=PhysicalCrash("wheel-hard-stop:stop",accepted.witnessed_contact_s,
        accepted.first_clear_s,accepted.witnessed_contact_s,0,
        accepted,prefix,version="wp3_grounded_wheel_hard_stop_crash_v4_dev")
    return crash

def test_explicit_development_source_and_real_rigid_impulse():
    c=event()
    assert SOURCES[c.version]==(
        "wheel_hard_stop","grounded_wheel_hard_stop_absorbing_v4_development")
    assert c.contact.kind=="wheel_hard_stop"
    assert c.contact.energy_lost_j==pytest.approx(20.)
    assert c.contact.kinetic_after_j==pytest.approx(0.)
    assert c.tick_contact_time_s<.1
    assert c.prefix[-1].start==c.prefix[-1].end
    assert c.contact.postimpact.position==c.contact.preimpact.position

@pytest.mark.parametrize("change",[
    {"obstacle_id":"road-patch:stop"},
    {"version":"wp3_grounded_applied_road_crash_v4_dev"},
    {"first_overlap_s":.0},
    {"last_proven_clear_s":.5},
])
def test_impersonation_of_road_or_obstacle_crash_is_rejected(change):
    with pytest.raises(ValueError):
        replace(event(),**change)
