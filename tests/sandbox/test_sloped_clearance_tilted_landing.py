"""WP7 bounded sloped support / swept road collision / low-tilt landing tests.

Only synthetic rectangles and time-step state; no HELD-OUT or PPO evidence.
"""
from dataclasses import replace
from math import sin, cos

import pytest

from f1rl.sandbox.ballistic_flight import (
    BallisticFlightPolicy, FlightPhysicsError, FlightState, advance_ballistic_flight,
)
from f1rl.sandbox.body_collision import VehicleBodyGeometry
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.independent_suspension_dynamics import (
    IndependentSuspensionDynamicsPolicy, SuspensionDynamicsState,
    SuspensionResolutionError,
)
from f1rl.sandbox.multi_surface_support import _atlas, MultiSurfaceSuspensionPolicy
from f1rl.sandbox.partial_wheel_release import AutomaticTakeoffPolicy, integrate_partial_support
from f1rl.sandbox.ramp_support import WheelContactPoints
from f1rl.sandbox.swept_surface_clearance import certify_swept_chassis_clearance
from f1rl.sandbox.track.patch import PlanarSurfacePatch
from f1rl.sandbox.vehicle import VehicleState, VehicleControl, VehicleParameters

BODY = VehicleBodyGeometry(1., .4, 2., Vec3(0., .1, 0.))
WHEELS = WheelContactPoints(
    Vec3(-.4,-.3,.8), Vec3(.4,-.3,.8),
    Vec3(-.4,-.3,-.8), Vec3(.4,-.3,-.8),
)
CONTACT = MultiSurfaceSuspensionPolicy(WHEELS, .12, .12)
PHYSICS = IndependentSuspensionDynamicsPolicy(CONTACT,1200.,1800.,600.,30000.,2000.,150.,
                                                integration_substeps=8)
SLOPE = .015
SLOPED_RELEASE = AutomaticTakeoffPolicy(.5,max_patch_tilt_rad=.04)
SLOPED_FLIGHT = BallisticFlightPolicy(CONTACT,6.,.12,max_patch_tilt_rad=.04)


def ramp(angle=SLOPE, *, name="ramp", length=4., center_z=0.):
    return PlanarSurfacePatch(
        name, Vec3(0., 0., center_z),
        Vec3(0., sin(angle), cos(angle)),
        Vec3(0., cos(angle), -sin(angle)),
        5., length,
    )


def vehicle(z=0., *, y=.3, vy=0., vz=0., pitch=0., roll=0., grounded=True):
    q = (Quaternion.from_axis_angle(Vec3(1.,0.,0.),pitch)
         * Quaternion.from_axis_angle(Vec3(0.,0.,1.),roll))
    return VehicleState(Vec3(0.,y,z),Vec3(0.,vy,vz),q,Vec3(0.,0.,0.),grounded)


def test_sloped_three_to_two_to_zero_contacts_and_release():
    start=vehicle(vz=15.)
    target=vehicle(z=2.9,vz=15.)
    dt=2.9/15.
    result=integrate_partial_support(
        start,target,SuspensionDynamicsState(.3),PHYSICS,BODY,
        (ramp(),),SLOPED_RELEASE,dt=dt,gravity_mps2=9.81,
    )
    assert result.takeoff_at_s is not None
    assert result.remaining_dt_s>0.
    assert not result.vehicle.grounded
    assert any(f.supported_wheels==2 for f in result.frames)
    assert result.vehicle.position.z>2.7


def test_slope_opt_in_is_strict_and_oversteep_ramp_blocked():
    start=vehicle()
    target=vehicle(z=.1)
    for policy in (AutomaticTakeoffPolicy(.5),
                   AutomaticTakeoffPolicy(.5,max_patch_tilt_rad=.01)):
        with pytest.raises(SuspensionResolutionError,match="road slope"):
            integrate_partial_support(
                start,target,SuspensionDynamicsState(.3),PHYSICS,BODY,
                (ramp(.08),),policy,dt=.02,gravity_mps2=9.81,
            )


def test_swept_chassis_catches_midframe_road_intrusion_even_if_endpoints_clear():
    plate=PlanarSurfacePatch(
        "raised",Vec3(0.,1.,0.),Vec3(0.,0.,1.),
        Vec3(0.,1.,0.),5.,1.,
    )
    atlas=_atlas((plate,),CONTACT.seam_height_tolerance_m)
    assert atlas is not None
    def pose(t):
        return vehicle(z=-5.+10.*t,y=.3)
    with pytest.raises(SuspensionResolutionError,match="penetration"):
        certify_swept_chassis_clearance(pose,0.,1.,BODY,atlas)


def test_clearance_accepts_safe_sloped_road_overhang():
    start=vehicle(z=1.,y=.3)
    end=vehicle(z=1.2,y=.3)
    atlas=_atlas((ramp(),),CONTACT.seam_height_tolerance_m)
    checked=certify_swept_chassis_clearance(
        lambda t: vehicle(z=1.+.2*t,y=.3),0.,1.,BODY,atlas,
    )
    assert checked>=1


def test_low_tilt_flat_touchdown_preserves_pitch_roll_in_suspension():
    start=vehicle(y=.9,vy=-1.,pitch=.03,roll=.025,grounded=False)
    flat=ramp(0.,length=10.)
    out=advance_ballistic_flight(
        start,SLOPED_FLIGHT,BODY,(flat,),(),dt=.5,gravity_mps2=9.81,
    )
    assert out.vehicle.grounded
    assert out.landing_suspension is not None
    assert out.landing_suspension.pitch_x_rad==pytest.approx(.03,abs=1e-8)
    assert out.landing_suspension.roll_z_rad==pytest.approx(.025,abs=1e-8)
    assert out.vehicle.orientation==start.orientation
    assert out.vehicle.velocity.y==0.


def test_inclined_touchdown_requires_registered_slope_and_uses_plane_speed():
    initial=vehicle(y=.9,vy=-1.,pitch=-SLOPE,grounded=False)
    with pytest.raises(FlightPhysicsError,match="sloped touchdown"):
        advance_ballistic_flight(
            initial,replace(SLOPED_FLIGHT,max_patch_tilt_rad=0.),BODY,
            (ramp(),),(),dt=.5,gravity_mps2=9.81,
        )
    out=advance_ballistic_flight(
        initial,SLOPED_FLIGHT,BODY,(ramp(),),(),
        dt=.5,gravity_mps2=9.81,
    )
    assert out.flight.phase=="grounded"
    assert out.landing_suspension is not None
    assert out.landing_suspension.pitch_x_rad==pytest.approx(-SLOPE,abs=1e-8)
    assert out.touchdown_speed_mps is not None and out.touchdown_speed_mps>0.


def test_landing_above_approved_tilt_is_rejected():
    start=vehicle(y=.9,vy=-1.,pitch=.25,grounded=False)
    with pytest.raises(FlightPhysicsError,match="touchdown tilt"):
        advance_ballistic_flight(
            start,SLOPED_FLIGHT,BODY,(ramp(0.,length=10.),),(),
            dt=.5,gravity_mps2=9.81,
        )


def test_world_landing_maintains_nonzero_attitude_through_remainder():
    from f1rl.sandbox.world import WorldState,step_world
    from f1rl.sandbox.interactions.state import InteractionState
    from f1rl.sandbox.rules.scoring import ScoreState
    from f1rl.sandbox.track.model import TrackWorld
    from f1rl.sandbox.track.state import TrackState

    start=vehicle(y=.9,vy=-1.,pitch=.025,roll=.015,grounded=False)
    world=WorldState(
        start,TrackState(frozenset()),
        InteractionState(frozenset(),frozenset()),ScoreState(),
        suspension=SuspensionDynamicsState(start.position.y),
        flight=FlightState("airborne"),
    )
    params=VehicleParameters(2.5,.3,.6,0.,20.,0.,0.,50.,9.81)
    result=step_world(
        TrackWorld(surfaces=(ramp(0.,length=10.),)),world,
        VehicleControl(0.,0.,0.),params,.5,.5,
        body_geometry=BODY,independent_suspension_policy=PHYSICS,
        ballistic_flight_policy=SLOPED_FLIGHT,
    )
    assert result.state.vehicle.grounded
    assert result.state.flight.phase=="grounded"
    assert result.state.suspension is not None
    assert abs(result.state.suspension.pitch_x_rad)>.001
    assert abs(result.state.suspension.roll_z_rad)>.001
    assert [event.kind for event in result.events]==["landing"]


@pytest.mark.parametrize("bad",[-.1,float("inf"),.5,True])
def test_invalid_slope_limits_fail_before_simulation(bad):
    with pytest.raises(ValueError):
        replace(SLOPED_FLIGHT,max_patch_tilt_rad=bad)
    with pytest.raises(ValueError):
        replace(SLOPED_RELEASE,max_patch_tilt_rad=bad)
