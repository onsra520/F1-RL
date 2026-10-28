"""Grounded yaw transfer and bounded lateral tire forces under opt-in policy."""
from dataclasses import replace
import pytest

from src.sandbox.grounded_yaw_dynamics import (
    GroundedYawPolicy,GroundedYawResolutionError,advance_grounded_yaw,
)
from src.sandbox.geometry.vector import Vec3
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.vehicle import VehicleState

POL=GroundedYawPolicy(
    1200.,1600.,1.25,1.25,18000.,18000.,.8,
    min_longitudinal_speed_mps=.5,
)

def grounded(*,forward=5.,lateral=0.,yaw_rate=0.):
    return VehicleState(
        Vec3(0.,.3,0.),Vec3(lateral,0.,forward),
        Quaternion.identity(),Vec3(0.,yaw_rate,0.),True,
    )

def tick(start=None,steering=0.,policy=POL,dt=.05):
    v=start or grounded()
    candidate=VehicleState(
        v.position,v.velocity,v.orientation,v.angular_velocity,True,
    )
    return advance_grounded_yaw(
        v,candidate,steering,policy,dt=dt,gravity_mps2=9.81,
    )


def test_straight_zero_yaw_does_not_generate_spurious_tire_force():
    after,audit=tick()
    assert after.angular_velocity.y==0.
    assert after.position.z==pytest.approx(.25)
    assert after.velocity.x==pytest.approx(0.)
    assert audit.front_force_n==0. and audit.rear_force_n==0.


def test_front_steering_generates_bounded_yaw_and_lateral_response():
    after,audit=tick(steering=.12)
    assert audit.front_slip_rad==pytest.approx(-.12)
    assert audit.rear_slip_rad==0.
    assert audit.front_force_n>0.
    assert audit.yaw_moment_nm>0.
    assert after.angular_velocity.y>0.
    assert after.position.x>0.
    assert after.grounded


def test_tire_force_clamped_by_registered_friction():
    policy=replace(POL,front_cornering_stiffness_nprad=2e6,
                   tire_friction_mu=.4)
    _,audit=tick(steering=.15,policy=policy)
    max_f=.4*1200.*9.81*.5
    assert audit.front_force_n==pytest.approx(max_f,abs=1e-6)


def test_nonzero_impact_yaw_rate_is_preserved_not_zeroed():
    initial=grounded(yaw_rate=.1)
    after,audit=tick(initial)
    assert after.angular_velocity.y!=0.
    assert audit.lateral_speed_before_mps==0.
    assert after.orientation!=initial.orientation


def test_low_speed_and_excessive_yaw_spin_fail_closed():
    with pytest.raises(GroundedYawResolutionError,match="low speed"):
        tick(grounded(forward=.1))
    with pytest.raises(GroundedYawResolutionError,match="incoming yaw spin"):
        tick(grounded(yaw_rate=2.))


@pytest.mark.parametrize("field,value",[
    ("mass_kg",0.),("yaw_inertia_kg_m2",float("nan")),
    ("tire_friction_mu",2.5),("max_slip_angle_rad",.9),
    ("front_cornering_stiffness_nprad",True),
])
def test_invalid_yaw_policy_rejected(field,value):
    with pytest.raises(ValueError):
        replace(POL,**{field:value})


def test_invalid_timestep_rejected():
    with pytest.raises(ValueError):
        tick(dt=0.)
