from dataclasses import replace
from math import atan2, cos, pi, sin, tan

import pytest

from src.research.wp7_track_feasibility import PARAMETERS
from src.sandbox.dynamics.ground_dynamics import compute_ground_motion
from src.sandbox.dynamics.integrator import integrate, integrate_ground_distance
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.geometry.vector import Vec3
from src.sandbox.surface import SurfaceContact
from src.sandbox.vehicle import VehicleControl, VehicleState

CONTACT = SurfaceContact(Vec3(0.,0.,0.),Vec3(0.,1.,0.))
PARAMS = replace(PARAMETERS,rolling_drag=0.,max_acceleration=4.,max_ground_speed=1.)


def state(speed):
    return VehicleState(Vec3(0.,.5,0.),Vec3(0.,0.,speed),Quaternion.identity(),Vec3(0.,0.,0.),True)


def step(previous,control,dt,parameters=PARAMS,contact=CONTACT):
    motion = compute_ground_motion(previous,control,parameters,contact)
    return integrate_ground_distance(previous,motion,control,parameters,contact,dt)


def test_speed_cap_event_integrates_pre_cap_and_capped_distances():
    end = step(state(.8),VehicleControl(1.,1.,0.),.1)
    distance = .8*.05+.5*4*.05**2+1.*.05
    curvature = tan(PARAMS.max_steering_angle)/PARAMS.wheelbase
    yaw = curvature*distance
    assert end.speed==pytest.approx(1.)
    assert end.position.x==pytest.approx((1-cos(yaw))/curvature)
    assert end.position.z==pytest.approx(sin(yaw)/curvature)
    assert end.angular_velocity.y==pytest.approx(curvature)


@pytest.mark.parametrize('control', (VehicleControl(1.,1.,0.),VehicleControl(-1.,0.,1.),VehicleControl(0.,0.,0.)))
def test_force_frozen_drag_free_step_agrees_with_two_half_steps(control):
    initial = state(.8)
    once = step(initial,control,.1)
    twice = step(step(initial,control,.05),control,.05)
    assert (once.position-twice.position).magnitude<1e-10
    assert (once.velocity-twice.velocity).magnitude<1e-10
    a,b = (s.orientation.rotate(Vec3(0.,0.,1.)) for s in (once,twice))
    assert abs(atan2(a.x,a.z)-atan2(b.x,b.z))<1e-10


@pytest.mark.parametrize('dt',(0.,-1.,float('nan'),float('inf')))
def test_invalid_tick_domain_rejected(dt):
    with pytest.raises(ValueError):
        step(state(0.),VehicleControl(0.,0.,0.),dt)


def test_nonhorizontal_contact_is_explicitly_unsupported():
    normal = Vec3(0.,cos(.1),-sin(.1))
    with pytest.raises(ValueError,match='horizontal contact'):
        step(state(.1),VehicleControl(0.,0.,0.),.1,contact=SurfaceContact(Vec3(0.,0.,0.),normal))


@pytest.mark.parametrize('velocity',(Vec3(.1,0.,.5),Vec3(0.,0.,-.1),Vec3(0.,0.,1.1)))
def test_unowned_lateral_reverse_or_overspeed_state_is_not_clamped(velocity):
    initial = replace(state(.5),velocity=velocity)
    with pytest.raises(ValueError,match='forward no-slip'):
        step(initial,VehicleControl(0.,0.,0.),.1)


def test_shortest_yaw_domain_rejects_ambiguous_multi_turn_tick():
    with pytest.raises(ValueError,match='shortest-yaw'):
        step(state(1.),VehicleControl(1.,0.,0.),20.)


def test_legacy_euler_entrypoint_retains_its_historical_numerical_law():
    initial = state(.5)
    control = VehicleControl(1.,0.,1.)
    motion = compute_ground_motion(initial,control,PARAMS,CONTACT)
    old = integrate(initial,motion,PARAMS,CONTACT,.1)
    assert old.position==initial.position
    old_heading = old.orientation.rotate(Vec3(0.,0.,1.))
    assert atan2(old_heading.x,old_heading.z)==pytest.approx(motion.angular_velocity.y*.1)
    corrected = step(initial,control,.1)
    assert corrected.position.z>initial.position.z
    assert corrected.angular_velocity.y==0.
