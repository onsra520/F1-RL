"""Bounded hard wheel-travel stop impulse and fail-closed crossing tests."""
from dataclasses import replace
import pytest
from f1rl.sandbox.wheel_bump_stop_impact import certify_wheel_bump_stop
from f1rl.sandbox.rigid_body_impact_response import RigidBodyImpactMass
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.vehicle import VehicleState
from f1rl.sandbox.track.patch import PlanarSurfacePatch

Q=Quaternion.identity()
Z=Vec3(0.,0.,0.)
UP=Vec3(0.,1.,0.)
ROAD=PlanarSurfacePatch("flat",Z,Vec3(0.,0.,1.),UP,20.,20.)
MASS=RigidBodyImpactMass(1000.,Vec3(500.,500.,500.))

def state(y,vy):
    return VehicleState(Vec3(0.,y,0.),Vec3(0.,vy,0.),Q,Z,True)


def kwargs():
    return {"wheel_local":Z,"road":ROAD,"max_bump_m":.12,
            "mass":MASS,"duration_s":.1}


def test_center_falling_into_bump_stop_creates_energy_losing_impulse():
    # gap start = 0.12 + (-0.11) = .01, end below hard stop.
    x=certify_wheel_bump_stop(state(-.11,-.2),state(-.13,-.2),**kwargs())
    assert .0499<x.first_clear_time_s<x.first_violating_time_s<.0501
    assert x.first_violating_time_s-x.first_clear_time_s<=1e-6
    assert x.impulse_n_s.y==pytest.approx(200.)
    assert x.angular_impulse_nms==Z
    assert x.kinetic_before_j==pytest.approx(20.)
    assert x.kinetic_after_j==pytest.approx(0.,abs=1e-7)
    assert x.dissipated_kinetic_j==pytest.approx(20.)
    assert x.normal_speed_before_mps<0. and abs(x.normal_speed_after_mps)<1e-8


def test_offcenter_stop_creates_bounded_pitch_roll_response():
    local=Vec3(.5,0.,0.)
    x=certify_wheel_bump_stop(state(-.11,-.2),state(-.13,-.2),
                              **{**kwargs(),"wheel_local":local})
    assert x.angular_impulse_nms.z>0.
    assert x.kinetic_after_j<=x.kinetic_before_j
    assert x.dissipated_kinetic_j>=0.


@pytest.mark.parametrize("a,b,v",[
    (-.11,-.10,-.2),(-.13,-.15,-.2),
    (-.11,-.13,.2),(-.13,-.11,.2)
])
def test_no_valid_incoming_crossing_fails_closed(a,b,v):
    with pytest.raises(ValueError):
        certify_wheel_bump_stop(state(a,v),state(b,v),**kwargs())


def test_elastic_impact_reverses_normal_motion_and_conserves_energy():
    x=certify_wheel_bump_stop(state(-.11,-.2),state(-.13,-.2),
        **{**kwargs(),"restitution":1.})
    assert x.normal_speed_after_mps==pytest.approx(.2)
    assert x.kinetic_after_j==pytest.approx(x.kinetic_before_j)


@pytest.mark.parametrize("bad",[-.01,float("nan"),float("inf"),True])
def test_invalid_hard_stop_distance_fails_closed(bad):
    with pytest.raises(ValueError):
        certify_wheel_bump_stop(state(-.11,-.2),state(-.13,-.2),
          **{**kwargs(),"max_bump_m":bad})
