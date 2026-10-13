"""Synthetic WP7 ballistic research. No TRAIN/HELD-OUT track access."""
from dataclasses import replace
from math import pi

import pytest

from f1rl.sandbox.ballistic_flight import (
    BallisticFlightPolicy, FlightPhysicsError, FlightState,
    authorize_takeoff, advance_ballistic_flight,
)
from f1rl.sandbox.body_collision import VehicleBodyGeometry
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.multi_surface_support import MultiSurfaceSuspensionPolicy
from f1rl.sandbox.ramp_support import WheelContactPoints
from f1rl.sandbox.track.patch import PlanarSurfacePatch
from f1rl.sandbox.track.volume import OrientedBoxVolume
from f1rl.sandbox.vehicle import VehicleState

BODY = VehicleBodyGeometry(1., .4, 2., Vec3(0., .1, 0.))
WHEELS = WheelContactPoints(
    Vec3(-.4, -.3, .8), Vec3(.4, -.3, .8),
    Vec3(-.4, -.3, -.8), Vec3(.4, -.3, -.8),
)
CONTACT = MultiSurfaceSuspensionPolicy(WHEELS, .12, .12)
POLICY = BallisticFlightPolicy(CONTACT, 3.0, .1)


def flat(name="flat", z=0., length=50., y=0.):
    return PlanarSurfacePatch(name, Vec3(0., y, z), Vec3(0., 0., 1.),
                              Vec3(0., 1., 0.), 10., length)


def car(y=.3, vy=2., z=0., vz=0., grounded=True,
        rotation=None, angular_velocity=None):
    return VehicleState(
        Vec3(0., y, z), Vec3(0., vy, vz),
        rotation or Quaternion.identity(),
        angular_velocity or Vec3(0., 0., 0.), grounded,
    )


def flight(vehicle, *, dt=.2, policy=POLICY, surfaces=None, obstacles=(), gravity=9.81):
    return advance_ballistic_flight(
        vehicle, policy, BODY, surfaces if surfaces is not None else (flat(),),
        obstacles, dt=dt, gravity_mps2=gravity,
    )


def test_explicit_supported_upward_launch_does_not_teleport():
    original = car()
    released = authorize_takeoff(original, POLICY, BODY, (flat(),))
    assert not released.grounded
    assert released.position == original.position
    assert released.velocity == original.velocity
    assert released.orientation == original.orientation


def test_unauthorized_downward_or_unverified_takeoff_rejected():
    for attempted in (car(vy=0.), car(vy=-1.), car(y=1.), car(grounded=False)):
        with pytest.raises(FlightPhysicsError):
            authorize_takeoff(attempted, POLICY, BODY, (flat(),))


def test_exact_ballistic_gravity_and_velocity_with_no_early_landing():
    released = authorize_takeoff(car(), POLICY, BODY, (flat(),))
    result = flight(released, dt=.2)
    assert result.flight.phase == "airborne"
    assert result.remaining_dt_s == 0.
    assert result.vehicle.position.y == pytest.approx(.3+2*.2-.5*9.81*.2**2)
    assert result.vehicle.velocity.y == pytest.approx(2.-9.81*.2)
    assert result.vehicle.position.z == pytest.approx(0.)
    assert all(not p.grounded for p in result.poses)


def test_deterministic_repeat_and_horizontal_motion():
    released = car(y=2., vy=1., z=-1., vz=4., grounded=False)
    a = flight(released, dt=.1)
    b = flight(released, dt=.1)
    assert a == b
    assert a.vehicle.position.z == pytest.approx(-.6)


def test_validated_touchdown_exact_time_and_residual_dt():
    released = authorize_takeoff(car(), POLICY, BODY, (flat(),))
    result = flight(released, dt=.5)
    impact_time = 4./9.81
    assert result.flight.phase == "grounded"
    assert result.vehicle.grounded
    assert result.vehicle.position.y == pytest.approx(.3, abs=1e-8)
    assert result.vehicle.velocity.y == 0.
    assert result.transitions[0].kind == "landing"
    assert result.transitions[0].time_s == pytest.approx(impact_time)
    assert result.remaining_dt_s == pytest.approx(.5-impact_time)
    assert result.touchdown_speed_mps == pytest.approx(2.)


def test_low_speed_limit_rejects_identical_trajectory():
    released = authorize_takeoff(car(), POLICY, BODY, (flat(),))
    with pytest.raises(FlightPhysicsError, match="vertical speed"):
        flight(released, dt=.5, policy=replace(POLICY, max_touchdown_vertical_speed_mps=1.))


def test_downward_impact_from_high_altitude_rejected():
    high = car(y=10., vy=-10., grounded=False)
    with pytest.raises(FlightPhysicsError, match="vertical speed"):
        flight(high, dt=1.)


def test_gap_has_no_false_road_at_expected_impact_time():
    released = car(y=1.5, vy=-1., grounded=False)
    far = flat("far", z=30., length=4.)
    result = flight(released, dt=.5, surfaces=(far,))
    assert result.flight.phase == "airborne"
    assert not result.vehicle.grounded


def test_partial_wheel_support_at_touchdown_cannot_be_counted_as_landing():
    # Narrow elevated strip can touch the left wheel only.
    narrow = PlanarSurfacePatch("strip", Vec3(-.4, 0., 0.),
                                Vec3(0., 0., 1.), Vec3(0., 1., 0.),
                                .4, 8.)
    released = car(y=1., vy=-2., grounded=False)
    with pytest.raises(FlightPhysicsError, match="touchdown road/wheel/body"):
        flight(released, dt=.4, surfaces=(narrow,))


def test_body_first_collision_precedes_wheels_and_is_rejected():
    # Fat hull bottom is below wheel reference points.
    oversized = VehicleBodyGeometry(1., 1.0, 2., Vec3(0., .1, 0.))
    released = car(y=1., vy=-1., grounded=False)
    with pytest.raises(FlightPhysicsError, match="chassis contacts road"):
        advance_ballistic_flight(released, POLICY, oversized,
                                 (flat(),), (), dt=.5, gravity_mps2=9.81)


def test_obstacle_between_fast_flight_endpoints_is_blocked():
    released = car(y=2., vy=0., z=-4., vz=80., grounded=False)
    thin = OrientedBoxVolume(Vec3(0., 1.8, 0.), Vec3(.5, .6, .02),
                            Quaternion.identity())
    with pytest.raises(FlightPhysicsError, match="obstacle sweep blocked"):
        flight(released, dt=.1, obstacles=(thin,))


def test_rotated_obstacle_is_not_assumed_clear():
    released = car(y=2., vy=0., grounded=False)
    rotated = OrientedBoxVolume(Vec3(0., 2., 3.), Vec3(1., 1., 1.),
                                Quaternion.from_axis_angle(Vec3(0., 1., 0.), pi/4.))
    with pytest.raises(FlightPhysicsError, match="rotated flight obstacle"):
        flight(released, obstacles=(rotated,))


def test_mixed_inclined_touchdown_unsupported():
    ramp = PlanarSurfacePatch(
        "ramp", Vec3(0., 0., 0.), Vec3(0., .1, .99498743710662),
        Vec3(0., .99498743710662, -.1), 10., 50.,
    )
    released = car(y=3., vy=-1., grounded=False)
    with pytest.raises(FlightPhysicsError, match="sloped touchdown"):
        flight(released, surfaces=(ramp,))


def test_overlapping_different_deck_elevations_fail_closed():
    released = car(y=3., vy=-1., grounded=False)
    with pytest.raises(FlightPhysicsError, match="stacked decks"):
        flight(released, surfaces=(flat("low"), flat("high", y=2.)))


def test_nonzero_airborne_angular_rate_unsupported():
    released = car(y=2., grounded=False,
                   angular_velocity=Vec3(0., .5, 0.))
    with pytest.raises(FlightPhysicsError, match="rotating airborne"):
        flight(released)


@pytest.mark.parametrize("value", [0., -1., float("inf"), True])
def test_invalid_touchdown_threshold(value):
    with pytest.raises(ValueError):
        replace(POLICY, max_touchdown_vertical_speed_mps=value)


@pytest.mark.parametrize("value", [1, 257, 1.5, True])
def test_invalid_collision_substeps(value):
    with pytest.raises(ValueError):
        replace(POLICY, collision_segments=value)


def test_takeoff_or_airborne_phase_is_explicitly_versioned():
    assert FlightState("grounded").phase == "grounded"
    assert FlightState("takeoff").phase == "takeoff"
    with pytest.raises(ValueError):
        FlightState("landed")
