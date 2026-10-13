"""Research-only platform edge / partial four-wheel support.

No model archives, TRAIN datasets, HELD-OUT files or PPO seeds are accessed.
"""
from dataclasses import replace
from math import pi

import pytest

from f1rl.sandbox.ballistic_flight import (
    BallisticFlightPolicy, FlightPhysicsError, FlightState,
)
from f1rl.sandbox.body_collision import VehicleBodyGeometry
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.independent_suspension_dynamics import (
    IndependentSuspensionDynamicsPolicy, SuspensionDynamicsState,
    SuspensionResolutionError,
)
from f1rl.sandbox.multi_surface_support import MultiSurfaceSuspensionPolicy
from f1rl.sandbox.partial_wheel_release import (
    AutomaticTakeoffPolicy, integrate_partial_support,
    sample_partial_wheels, _flat_atlas,
)
from f1rl.sandbox.ramp_support import WheelContactPoints
from f1rl.sandbox.track.patch import PlanarSurfacePatch
from f1rl.sandbox.vehicle import VehicleState, VehicleControl, VehicleParameters


BODY = VehicleBodyGeometry(1., .4, 2., Vec3(0., .1, 0.))
WHEELS = WheelContactPoints(
    Vec3(-.4, -.3, .8), Vec3(.4, -.3, .8),
    Vec3(-.4, -.3, -.8), Vec3(.4, -.3, -.8),
)
CONTACT = MultiSurfaceSuspensionPolicy(WHEELS, .12, .12)
PHYSICS = IndependentSuspensionDynamicsPolicy(
    CONTACT, 1200., 1800., 600., 30000., 2000., 150.,
    integration_substeps=8,
)
POLICY = AutomaticTakeoffPolicy(max_substep_xz_m=.5)
FLIGHT = BallisticFlightPolicy(CONTACT, 6., .15)


def road(z=0., length=4., width=5., name="platform", elevation=0.):
    return PlanarSurfacePatch(name, Vec3(0., elevation, z),
                              Vec3(0., 0., 1.), Vec3(0., 1., 0.),
                              width, length)


def pose(z=0., *, y=.3, speed=0., rotation=None):
    return VehicleState(Vec3(0., y, z), Vec3(0., 0., speed),
                        rotation or Quaternion.identity(),
                        Vec3(0., 0., 0.), True)


def integrate(z=0., *, travel=0., dt=.4, start=None,
              surfaces=None, auto=POLICY):
    p = start or pose(z, speed=travel/dt)
    target = pose(p.position.z + travel, speed=p.velocity.z)
    return integrate_partial_support(
        p, target, SuspensionDynamicsState(p.position.y),
        PHYSICS, BODY,
        surfaces if surfaces is not None else (road(),),
        auto, dt=dt, gravity_mps2=9.81,
    )


def test_four_supported_wheels_have_equal_rest_load():
    report = integrate(0., dt=.04)
    assert report.takeoff_at_s is None
    assert report.vehicle.grounded
    assert report.frames[0].supported_wheels == 4
    assert report.frames[0].forces_n == pytest.approx((2943.,)*4)
    assert report.vehicle.position.y == pytest.approx(.3)


def test_front_axle_leaves_platform_rear_axle_remains_loaded():
    # At z=1.3 the front wheel z=2.1 is already beyond platform end=2.0.
    p = pose(1.3)
    atlas = _flat_atlas((road(),), CONTACT.seam_height_tolerance_m)
    wheels = sample_partial_wheels(p, CONTACT, atlas)
    assert wheels[:2] == (None, None)
    assert all(c is not None for c in wheels[2:])
    result = integrate(start=p, dt=.02)
    assert result.takeoff_at_s is None
    assert result.frames[0].supported_wheels == 2
    assert result.frames[0].forces_n[:2] == (0., 0.)
    assert all(n > 0. for n in result.frames[0].forces_n[2:])


def test_auto_release_only_when_all_four_wheels_exit_with_substep_time():
    p = pose(0., speed=15.)
    out = integrate(start=p, travel=2.9, dt=2.9/15.)
    assert out.takeoff_at_s is not None
    assert .16 < out.takeoff_at_s < 2.9/15.
    assert not out.vehicle.grounded
    assert out.vehicle.position.z > 2.8
    assert out.remaining_dt_s == pytest.approx(2.9/15.-out.takeoff_at_s)
    assert any(f.supported_wheels == 2 for f in out.frames)
    assert any(f.forces_n[0] == 0. and f.forces_n[2] > 0. for f in out.frames)
    assert out.vehicle.angular_velocity.magnitude > 0.
    atlas = _flat_atlas((road(),), CONTACT.seam_height_tolerance_m)
    assert sample_partial_wheels(out.vehicle, CONTACT, atlas) == (None,)*4


def test_reversed_input_is_exactly_repeatable():
    p = pose(0., speed=15.)
    assert integrate(start=p, travel=2.9, dt=2.9/15.) == integrate(
        start=p, travel=2.9, dt=2.9/15.,
    )


def test_no_launch_just_for_one_or_two_missing_contacts():
    result = integrate(start=pose(1.4), dt=.02)
    assert result.vehicle.grounded
    assert result.takeoff_at_s is None


def test_no_support_at_initial_grounded_pose_is_invalid():
    with pytest.raises(SuspensionResolutionError, match="zero supported"):
        integrate(start=pose(5.), travel=0., dt=.1)


def test_excessive_droop_is_not_misdiagnosed_as_platform_edge():
    with pytest.raises(SuspensionResolutionError, match="exceeds droop"):
        integrate(start=pose(0., y=1.), dt=.02)


def test_bump_penetration_not_inferred_as_successful_support():
    with pytest.raises(SuspensionResolutionError, match="penetrates road"):
        integrate(start=pose(0., y=.1), dt=.02)


def test_horizontal_resolution_limit_is_fail_closed():
    with pytest.raises(SuspensionResolutionError, match="XZ movement"):
        integrate(start=pose(0., speed=20.), travel=4.8, dt=.2)


def test_sloped_platform_requires_separate_support_transition_policy():
    incline = PlanarSurfacePatch(
        "incline", Vec3(0., 0., 0.), Vec3(0., .1, .99498743710662),
        Vec3(0., .99498743710662, -.1), 5., 8.,
    )
    with pytest.raises(SuspensionResolutionError, match="flat roads"):
        integrate(surfaces=(incline,), dt=.04)


def test_discontinuous_overlapping_decks_fail_closed():
    with pytest.raises(SuspensionResolutionError, match="road atlas"):
        integrate(surfaces=(road(name="bottom"), road(name="top", elevation=2.)), dt=.04)


@pytest.mark.parametrize("options", [
    {"max_substep_xz_m": 0.},
    {"max_substep_xz_m": float("nan")},
    {"release_time_tolerance_s": True},
    {"release_bisections": 3},
    {"release_bisections": 49},
])
def test_bad_release_policy_inputs_rejected(options):
    with pytest.raises(ValueError):
        replace(POLICY, **options)


def _world(vehicle):
    from f1rl.sandbox.interactions.state import InteractionState
    from f1rl.sandbox.rules.scoring import ScoreState
    from f1rl.sandbox.track.model import TrackWorld
    from f1rl.sandbox.track.state import TrackState
    from f1rl.sandbox.world import WorldState
    world = WorldState(
        vehicle, TrackState(frozenset()),
        InteractionState(frozenset(), frozenset()), ScoreState(),
        suspension=SuspensionDynamicsState(vehicle.position.y),
        flight=FlightState("grounded"),
    )
    params = VehicleParameters(2.5,.3,.6,0.,20.,0.,0.,50.,9.81)
    return TrackWorld(surfaces=(road(),)), world, params


def test_world_auto_takeoff_switches_to_ballistic_for_remaining_time():
    from f1rl.sandbox.world import step_world
    source = pose(speed=15.)
    track, state, parameters = _world(source)
    result = step_world(
        track, state, VehicleControl(0.,0.,0.), parameters, 2.9/15.,.5,
        body_geometry=BODY, independent_suspension_policy=PHYSICS,
        ballistic_flight_policy=FLIGHT, automatic_takeoff_policy=POLICY,
    )
    assert result.partial_support_telemetry is not None
    assert result.partial_support_telemetry.takeoff_at_s is not None
    assert result.flight_telemetry is not None
    assert result.state.flight.phase == "airborne"
    assert not result.state.vehicle.grounded
    assert len(result.events) == 1
    assert result.events[0].kind == "takeoff"
    assert result.events[0].time_s == pytest.approx(
        result.partial_support_telemetry.takeoff_at_s,
    )
    assert result.state.vehicle.position.z == pytest.approx(2.9,abs=1e-7)
    assert state.vehicle.grounded
    assert state.flight.phase == "grounded"


def test_world_no_automatic_policy_retains_legacy_optin_path():
    from f1rl.sandbox.world import step_world
    source = pose()
    track, state, parameters = _world(source)
    result = step_world(
        track, state, VehicleControl(0.,0.,0.), parameters,.02,.5,
        body_geometry=BODY, independent_suspension_policy=PHYSICS,
        ballistic_flight_policy=FLIGHT,
    )
    assert result.partial_support_telemetry is None
    assert result.state.flight.phase == "grounded"
    assert result.state.vehicle.grounded


def test_world_refuses_auto_policy_without_complete_optin():
    from f1rl.sandbox.world import step_world
    source = pose()
    track, state, parameters = _world(source)
    with pytest.raises(ValueError, match="automatic_takeoff_policy requires"):
        step_world(
            track, state, VehicleControl(0.,0.,0.), parameters,.02,.5,
            body_geometry=BODY,
            independent_suspension_policy=PHYSICS,
            automatic_takeoff_policy=POLICY,
        )


def test_world_auto_departure_cannot_collect_airborne_reward():
    from f1rl.sandbox.entities.score_pickup import ScorePickup
    from f1rl.sandbox.track.volume import OrientedBoxVolume
    from f1rl.sandbox.track.model import TrackWorld
    from f1rl.sandbox.world import step_world
    source = pose(speed=15.)
    track, state, parameters = _world(source)
    volume = OrientedBoxVolume(Vec3(0.,.3,2.9),Vec3(2.,2.,1.),
                                Quaternion.identity())
    track = TrackWorld(surfaces=track.surfaces,
                       score_pickups=(ScorePickup("unearned",volume,99),))
    result = step_world(
        track, state, VehicleControl(0.,0.,0.), parameters, 2.9/15.,.5,
        body_geometry=BODY, independent_suspension_policy=PHYSICS,
        ballistic_flight_policy=FLIGHT, automatic_takeoff_policy=POLICY,
    )
    assert result.state.score == state.score
    assert result.state.track == state.track
    assert [e.kind for e in result.events] == ["takeoff"]


def test_world_keeps_same_auto_policy_during_subsequent_airborne_frame():
    from f1rl.sandbox.world import step_world
    track, state, parameters = _world(pose(speed=15.))
    opts = dict(
        body_geometry=BODY,
        independent_suspension_policy=PHYSICS,
        ballistic_flight_policy=FLIGHT,
        automatic_takeoff_policy=POLICY,
    )
    first = step_world(
        track, state, VehicleControl(0.,0.,0.), parameters,2.9/15.,.5,**opts,
    )
    assert first.state.flight.phase == "airborne"
    second = step_world(
        track, first.state, VehicleControl(0.,0.,0.), parameters,.01,.5,**opts,
    )
    assert second.state.flight.phase == "airborne"
    assert second.partial_support_telemetry is None
    assert second.state.vehicle.position.z > first.state.vehicle.position.z
    assert second.events == ()


def test_fast_angular_motion_cannot_alias_shortest_arc_sweep():
    from f1rl.sandbox.ballistic_flight import advance_ballistic_flight
    spinning = VehicleState(
        Vec3(0.,10.,5.), Vec3(0.,0.,300.),
        Quaternion.identity(), Vec3(0.,100.,0.), False,
    )
    with pytest.raises(FlightPhysicsError, match="angular sweep resolution"):
        advance_ballistic_flight(
            spinning, FLIGHT, BODY, (road(),), (), dt=.2, gravity_mps2=9.81,
        )
