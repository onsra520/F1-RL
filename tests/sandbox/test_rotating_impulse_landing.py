"""Opt-in rotating touchdown and audited single-wheel impulse integration.

Synthetic rectangular patches only; not a full 6-DOF contact manifold or
an accepted WP7 calibration protocol.
"""
from dataclasses import replace

import pytest

from f1rl.sandbox.angular_landing_impulse import AngularLandingImpulsePolicy
from f1rl.sandbox.ballistic_flight import (
    BallisticFlightPolicy, FlightPhysicsError, FlightState,
    advance_ballistic_flight,
)
from f1rl.sandbox.body_collision import VehicleBodyGeometry
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.independent_suspension_dynamics import (
    IndependentSuspensionDynamicsPolicy, SuspensionDynamicsState,
)
from f1rl.sandbox.multi_surface_support import MultiSurfaceSuspensionPolicy
from f1rl.sandbox.ramp_support import WheelContactPoints
from f1rl.sandbox.track.patch import PlanarSurfacePatch
from f1rl.sandbox.vehicle import VehicleState, VehicleControl, VehicleParameters

BODY=VehicleBodyGeometry(1.,.4,2.,Vec3(0.,.1,0.))
WHEELS=WheelContactPoints(
    Vec3(-.4,-.3,.8),Vec3(.4,-.3,.8),
    Vec3(-.4,-.3,-.8),Vec3(.4,-.3,-.8),
)
CONTACT=MultiSurfaceSuspensionPolicy(WHEELS,.12,.12)
PHYSICS=IndependentSuspensionDynamicsPolicy(CONTACT,1200.,1800.,600.,30000.,2000.,150.,
                                             integration_substeps=8)
IMPULSE=AngularLandingImpulsePolicy(restitution=0.)
FLIGHT=BallisticFlightPolicy(CONTACT,6.,.12,angular_impulse=IMPULSE)


def plane(width=10.,length=20.,angle=0.):
    from math import sin,cos
    return PlanarSurfacePatch(
        "landing",Vec3(0.,0.,0.),Vec3(0.,sin(angle),cos(angle)),
        Vec3(0.,cos(angle),-sin(angle)),width,length,
    )


def airborne(*, y=.85, vy=-1., pitch=.025,roll=.03, wx=.1, wz=0.):
    rotation=(Quaternion.from_axis_angle(Vec3(1.,0.,0.),pitch)
              *Quaternion.from_axis_angle(Vec3(0.,0.,1.),roll))
    return VehicleState(
        Vec3(0.,y,0.),Vec3(1.,vy,0.),
        rotation,Vec3(wx,0.,wz),False,
    )


def fly(state=None,*, dt=.25, policy=FLIGHT, surfaces=None, dynamics=PHYSICS):
    return advance_ballistic_flight(
        state or airborne(),policy,BODY,surfaces or (plane(),),(),
        dt=dt,gravity_mps2=9.81,landing_dynamics=dynamics,
    )


def test_one_wheel_rotating_touchdown_has_explicit_energy_audit_and_persistent_rates():
    result=fly()
    assert result.vehicle.grounded
    assert result.flight.phase=="grounded"
    assert result.impulse_audit is not None
    audit=result.impulse_audit
    assert audit.magnitude_ns>0.
    assert audit.post_kinetic_energy_j<audit.pre_kinetic_energy_j
    assert audit.normal_speed_after_mps==pytest.approx(0.,abs=1e-7)
    assert result.landing_suspension is not None
    assert abs(result.landing_suspension.pitch_x_rad)>.01
    assert abs(result.landing_suspension.roll_z_rad)>.01
    assert abs(result.landing_suspension.pitch_rate_radps)>0.
    assert abs(result.landing_suspension.roll_rate_radps)>0.
    assert result.vehicle.angular_velocity.magnitude>0.
    assert result.remaining_dt_s>0.


def test_zero_rotation_follows_existing_landing_without_impulse():
    state=airborne(wx=0.)
    result=fly(state)
    assert result.vehicle.grounded
    assert result.impulse_audit is None
    assert result.landing_suspension.pitch_rate_radps==0.


def test_rotating_in_flight_without_touchdown_remains_ballistic():
    state=airborne(y=3.,vy=.1)
    result=fly(state,dt=.05)
    assert result.flight.phase=="airborne"
    assert result.impulse_audit is None
    assert not result.vehicle.grounded


def test_opt_in_without_explicit_inertia_rejected():
    with pytest.raises(FlightPhysicsError,match="requires explicit impulse inertia"):
        fly(dynamics=None)


def test_simultaneous_symmetric_wheel_contact_is_unsupported():
    state=airborne(pitch=0.,roll=0.,wx=.05)
    # Positive X rotation makes front wheels lower together.
    with pytest.raises(FlightPhysicsError,match="simultaneous wheel contact manifold"):
        fly(state)


def test_rotating_sloped_landing_not_misreported_as_accepted_impulse():
    inclined=plane(angle=.02)
    policy=replace(FLIGHT,max_patch_tilt_rad=.03)
    with pytest.raises(FlightPhysicsError,match="sloped rotating touchdown"):
        fly(policy=policy,surfaces=(inclined,))


def test_restution_one_never_increases_kinetic_energy():
    result=fly(policy=replace(FLIGHT,angular_impulse=replace(IMPULSE,restitution=1.)))
    assert result.impulse_audit.post_kinetic_energy_j==pytest.approx(
        result.impulse_audit.pre_kinetic_energy_j,rel=1e-10,
    )


def test_world_preserves_rate_state_after_accepted_angular_touchdown():
    from f1rl.sandbox.interactions.state import InteractionState
    from f1rl.sandbox.rules.scoring import ScoreState
    from f1rl.sandbox.track.model import TrackWorld
    from f1rl.sandbox.track.state import TrackState
    from f1rl.sandbox.world import WorldState,step_world

    state=airborne()
    world=WorldState(
        state,TrackState(frozenset()),
        InteractionState(frozenset(),frozenset()),ScoreState(),
        suspension=SuspensionDynamicsState(state.position.y),
        flight=FlightState("airborne"),
    )
    params=VehicleParameters(2.5,.3,.6,0.,20.,0.,0.,50.,9.81)
    result=step_world(
        TrackWorld(surfaces=(plane(),)),world,
        VehicleControl(0.,0.,0.),params,.25,.5,
        body_geometry=BODY,independent_suspension_policy=PHYSICS,
        ballistic_flight_policy=FLIGHT,
    )
    assert result.flight_telemetry is not None
    assert result.flight_telemetry.impulse_audit is not None
    assert result.state.flight.phase=="grounded"
    assert result.state.vehicle.grounded
    assert result.state.suspension is not None
    assert abs(result.state.suspension.pitch_rate_radps)>.01
    assert result.state.score==world.score
    assert [x.kind for x in result.events]==["landing"]
