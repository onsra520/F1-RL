"""Synthetic anti-tunnelling for certified piecewise-RK4 / ballistic sweeps.

The trajectory is a versioned numerical model: RK4 angular anchors with
shortest-arc interpolation. It does not certify analytic Euler ODE
integration error or accept 6DOF impact resolution.
"""
from dataclasses import replace
from math import pi
import pytest

from src.sandbox.ballistic_flight import (
    BallisticFlightPolicy,FlightPhysicsError,FlightState,advance_ballistic_flight,
)
from src.sandbox.body_collision import VehicleBodyGeometry
from src.sandbox.contact_manifold import ContactManifoldPolicy
from src.sandbox.geometry.vector import Vec3
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.independent_suspension_dynamics import (
    IndependentSuspensionDynamicsPolicy,SuspensionDynamicsState,
)
from src.sandbox.multi_surface_support import MultiSurfaceSuspensionPolicy
from src.sandbox.ramp_support import WheelContactPoints
from src.sandbox.torque_free_rotation import TorqueFreeRotationPolicy
from src.sandbox.torque_free_swept_ccd import TorqueFreeSweepPolicy
from src.sandbox.track.patch import PlanarSurfacePatch
from src.sandbox.track.volume import OrientedBoxVolume
from src.sandbox.vehicle import VehicleState,VehicleControl,VehicleParameters

BODY=VehicleBodyGeometry(1.,.4,2.,Vec3(0.,.1,0.))
W=WheelContactPoints(
    Vec3(-.4,-.3,.8),Vec3(.4,-.3,.8),
    Vec3(-.4,-.3,-.8),Vec3(.4,-.3,-.8),
)
CONTACT=MultiSurfaceSuspensionPolicy(W,.12,.12)
PHYS=IndependentSuspensionDynamicsPolicy(
    CONTACT,1200.,1800.,600.,30000.,2000.,150.,integration_substeps=8,
)
MANIFOLD=ContactManifoldPolicy(yaw_inertia_kg_m2=1600.)
FREE=TorqueFreeRotationPolicy(1800.,1600.,600.,integration_steps=16)
SWEEP=TorqueFreeSweepPolicy(slices=16,max_subdivisions=10)
FLIGHT=BallisticFlightPolicy(
    CONTACT,6.,.12,manifold=MANIFOLD,torque_free=FREE,
    torque_free_sweep=SWEEP,
)
ROAD=PlanarSurfacePatch(
    "low",Vec3(0.,0.,0.),Vec3(0.,0.,1.),Vec3(0.,1.,0.),20.,30.,
)


def car(*,p=Vec3(0.,5.,0.),v=Vec3(1.,1.,0.),
        w=Vec3(.15,.24,.17)):
    return VehicleState(
        p,v,Quaternion.identity(),w,False,
    )


def obstacle(p=Vec3(4.,5.,0.),half=Vec3(.5,.5,.5),yaw=0.):
    return OrientedBoxVolume(
        p,half,Quaternion.from_axis_angle(Vec3(0.,1.,0.),yaw),
    )


def run(initial=None, *, roads=(ROAD,),obstacles=(),dt=.2,policy=FLIGHT):
    return advance_ballistic_flight(
        initial or car(),policy,BODY,roads,obstacles,
        dt=dt,gravity_mps2=9.81,landing_dynamics=PHYS,
    )


def test_obstacle_far_but_present_is_certified_under_6dof_precession():
    initial=car()
    out=run(initial,obstacles=(obstacle(),))
    assert out.flight.phase=="airborne"
    assert out.angular_audit is not None
    assert len(out.poses)==SWEEP.slices+1
    assert out.vehicle.orientation!=initial.orientation
    assert out.vehicle.angular_velocity!=initial.angular_velocity
    assert out.transitions==()
    assert out.vehicle.position.x==pytest.approx(.2)


def test_rotated_obstacle_clear_is_certified():
    out=run(obstacles=(obstacle(yaw=pi/4.),))
    assert out.flight.phase=="airborne"
    assert out.angular_audit is not None


def test_fast_midstep_obstacle_collision_cannot_tunnel_between_endpoints():
    initial=car(p=Vec3(-5.,5.,0.),v=Vec3(50.,0.,0.),w=Vec3(0.,0.,0.))
    with pytest.raises(FlightPhysicsError,match="body/obstacle CCD rejected"):
        run(initial,obstacles=(obstacle(Vec3(0.,5.,0.),Vec3(.3,.5,.3)),))


def test_rotated_obstacle_midstep_collision_is_detected():
    initial=car(p=Vec3(-5.,5.,0.),v=Vec3(50.,0.,0.))
    with pytest.raises(FlightPhysicsError,match="body/obstacle CCD rejected"):
        run(initial,obstacles=(obstacle(Vec3(0.,5.,0.),Vec3(.3,.5,.3),pi/4.),))


def test_low_altitude_hover_is_safe_even_when_bounding_sphere_fails():
    initial=car(p=Vec3(0.,1.1,0.),v=Vec3(0.,0.,0.),w=Vec3(0.,0.,0.))
    out=run(initial,dt=.1)
    assert out.flight.phase=="airborne"
    assert out.angular_audit is not None
    assert out.vehicle.position.y<1.1


def test_bump_in_middle_of_xz_path_blocks_while_ends_outside():
    raised=PlanarSurfacePatch(
        "raised",Vec3(0.,1.02,0.),Vec3(0.,0.,1.),
        Vec3(0.,1.,0.),1.,.4,
    )
    initial=car(p=Vec3(-4.,1.1,0.),v=Vec3(40.,0.,0.),
                w=Vec3(0.,0.,0.))
    with pytest.raises(FlightPhysicsError,match="chassis/road CCD rejected"):
        run(initial,roads=(raised,),dt=.2)


def test_wheel_road_crossing_blocks_even_when_chassis_has_positive_clearance():
    initial=car(p=Vec3(0.,.34,0.),v=Vec3(0.,0.,0.),
                w=Vec3(0.,0.,0.))
    with pytest.raises(FlightPhysicsError,match="wheel"):
        run(initial,dt=.1)


def test_initial_obb_contact_blocked_without_reward():
    with pytest.raises(FlightPhysicsError,match="body/obstacle CCD rejected"):
        run(obstacles=(obstacle(Vec3(0.,5.,0.),Vec3(.2,.3,.2)),))


def test_insufficient_rotational_sampling_is_fail_closed():
    initial=car(w=Vec3(0.,8.,0.))
    policy=replace(
        FLIGHT,torque_free_sweep=replace(
            SWEEP,slices=4,max_slice_rotation_rad=.05,
        ),
    )
    with pytest.raises(FlightPhysicsError,match="anchor rotation too large"):
        run(initial,dt=.5,policy=policy)


def test_obstacle_query_and_replay_are_deterministic():
    first=run(obstacles=(obstacle(),))
    again=run(obstacles=(obstacle(),))
    assert first==again


def test_opt_in_requires_registered_angular_policy():
    with pytest.raises(ValueError,match="torque_free_sweep requires"):
        replace(FLIGHT,torque_free=None)


@pytest.mark.parametrize("field,value",[
    ("slices",0),("slices",257),("slices",True),
    ("max_subdivisions",21),("max_subdivisions",True),
    ("max_slice_rotation_rad",0.),("max_slice_rotation_rad",.5),
    ("clearance_epsilon_m",float("nan")),("clearance_epsilon_m",.02),
])
def test_invalid_ccd_config_rejected(field,value):
    with pytest.raises(ValueError):
        replace(SWEEP,**{field:value})


def test_world_step_continues_angular_flight_without_phantom_events():
    from src.sandbox.interactions.state import InteractionState
    from src.sandbox.rules.scoring import ScoreState
    from src.sandbox.track.model import TrackWorld
    from src.sandbox.track.state import TrackState
    from src.sandbox.world import WorldState,step_world

    initial=car()
    current=WorldState(
        initial,TrackState(frozenset()),
        InteractionState(frozenset(),frozenset()),ScoreState(),
        suspension=SuspensionDynamicsState(initial.position.y),
        flight=FlightState("airborne"),
    )
    params=VehicleParameters(2.5,.3,.6,0.,20.,0.,0.,50.,9.81)
    result=step_world(
        TrackWorld(surfaces=(ROAD,)),current,
        VehicleControl(0.,0.,0.),params,.2,.5,
        body_geometry=BODY,independent_suspension_policy=PHYS,
        ballistic_flight_policy=FLIGHT,
    )
    assert result.state.flight.phase=="airborne"
    assert result.state.vehicle.angular_velocity!=initial.angular_velocity
    assert result.flight_telemetry.angular_audit is not None
    assert result.state.score==current.score
    assert result.events==()
