"""Analytic Euler torque-free clearance enclosure vs numerical RK4 CCD.

Every 'clear' result follows a conservative energy-derived angular-speed
bound, NOT trust in RK4 accuracy. Unresolved intervals are not contacts.
"""
from dataclasses import replace
import pytest

from f1rl.sandbox.euler_ode_clearance import (
    EulerODEEnclosurePolicy,certify_torque_free_ode_clearance,
)
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.body_collision import VehicleBodyGeometry
from f1rl.sandbox.multi_surface_support import _atlas
from f1rl.sandbox.ramp_support import WheelContactPoints
from f1rl.sandbox.track.patch import PlanarSurfacePatch
from f1rl.sandbox.track.volume import OrientedBoxVolume
from f1rl.sandbox.vehicle import VehicleState
from f1rl.sandbox.torque_free_rotation import TorqueFreeRotationPolicy
from f1rl.sandbox.torque_free_swept_ccd import TorqueFreeSweepPolicy

ROAD=PlanarSurfacePatch("plane",Vec3(0.,0.,0.),Vec3(0.,0.,1.),Vec3(0.,1.,0.),10.,20.)
ATLAS=_atlas((ROAD,),1e-6)
BODY=VehicleBodyGeometry(1.,.4,2.,Vec3(0.,.1,0.))
W=WheelContactPoints(
    Vec3(-.4,-.3,.8),Vec3(.4,-.3,.8),
    Vec3(-.4,-.3,-.8),Vec3(.4,-.3,-.8),
)
ANGULAR=TorqueFreeRotationPolicy(1800.,1600.,600.,integration_steps=32)
POLICY=EulerODEEnclosurePolicy(max_subdivisions=12)


def airborne(*,pos=Vec3(0.,5.,0.),vel=Vec3(1.,0.,0.),
             spin=Vec3(.15,.24,.17)):
    return VehicleState(pos,vel,Quaternion.identity(),spin,False)

def prove(start=None, *, obstacles=(),duration=.2,policy=POLICY):
    return certify_torque_free_ode_clearance(
        start or airborne(),ANGULAR,BODY,W,ATLAS,obstacles,
        duration_s=duration,gravity_mps2=9.81,policy=policy,
    )

def box(pos=Vec3(5.,5.,0.)):
    return OrientedBoxVolume(pos,Vec3(.2,.3,.2),Quaternion.identity())

def test_far_road_and_obstacle_proved_clear_despite_unknown_rk4_error():
    v=prove(obstacles=(box(),))
    assert v.valid
    assert v.first_uncertain_window_s is None
    assert v.angular_speed_bound_radps>0.
    assert v.checks>0.

def test_fast_midstep_obstacle_tunnel_cannot_pass_analytic_enclosure():
    start=airborne(pos=Vec3(-5.,5.,0.),vel=Vec3(50.,0.,0.),
                   spin=Vec3(0.,0.,0.))
    v=prove(start,obstacles=(box(Vec3(0.,5.,0.)),))
    assert not v.valid
    assert v.status=="uncertain"
    lo,hi=v.first_uncertain_window_s
    assert lo<.1<hi or abs(lo-.1)<.02

def test_road_wheel_proximity_is_uncertain_never_falsely_safe():
    near=airborne(pos=Vec3(0.,.3,0.),vel=Vec3(0.,0.,0.),
                  spin=Vec3(0.,0.,0.))
    v=prove(near,duration=.05)
    assert not v.valid
    assert v.first_uncertain_window_s is not None

def test_low_clearance_but_static_body_above_road_can_be_proved():
    still=airborne(pos=Vec3(0.,1.1,0.),vel=Vec3(0.,0.,0.),
                   spin=Vec3(0.,0.,0.))
    result=prove(still,duration=.1)
    assert result.valid

def test_increasing_angular_speed_does_not_shrink_analytic_uncertainty():
    slow=airborne(pos=Vec3(0.,1.1,0.),vel=Vec3(0.,0.,0.),
                  spin=Vec3(0.,0.,0.))
    fast=airborne(pos=slow.position,vel=slow.velocity,
                  spin=Vec3(0.,6.,0.))
    a=prove(slow,duration=.1)
    b=prove(fast,duration=.1)
    assert a.valid and not b.valid
    assert b.angular_speed_bound_radps>a.angular_speed_bound_radps

def test_optin_enclosure_policy_is_separate_from_piecewise_ccd():
    sweep=TorqueFreeSweepPolicy(analytic_ode=POLICY)
    assert sweep.analytic_ode is POLICY
    with pytest.raises(ValueError,match="analytic_ode"):
        replace(sweep,analytic_ode="unsafe")

@pytest.mark.parametrize("field,value",[
    ("max_subdivisions",-1),("max_subdivisions",21),
    ("max_subdivisions",True),("min_clearance_m",float("nan")),
    ("min_clearance_m",0.),("min_clearance_m",.5),
])
def test_invalid_enclosure_policy_rejected(field,value):
    with pytest.raises(ValueError):
        replace(POLICY,**{field:value})
