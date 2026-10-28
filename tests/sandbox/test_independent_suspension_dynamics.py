"""Synthetic 3-DOF independent suspension dynamics, no scientific tracks."""
from dataclasses import replace
from math import pi, sin, cos

import pytest

from src.simulation.body_collision import VehicleBodyGeometry
from src.simulation.geometry.rotation import Quaternion
from src.simulation.geometry.vector import Vec3
from src.simulation.independent_suspension_dynamics import (
    IndependentSuspensionDynamicsPolicy, SuspensionDynamicsState,
    SuspensionResolutionError, integrate_independent_suspension,
    suspension_orientation,
)
from src.simulation.multi_surface_support import MultiSurfaceSuspensionPolicy
from src.simulation.ramp_support import WheelContactPoints
from src.simulation.track.patch import PlanarSurfacePatch
from src.simulation.vehicle import VehicleState

BODY = VehicleBodyGeometry(1., .4, 2., Vec3(0., .1, 0.))
WHEELS = WheelContactPoints(Vec3(-.4, -.3, .8), Vec3(.4, -.3, .8),
                            Vec3(-.4, -.3, -.8), Vec3(.4, -.3, -.8))
CONTACT = MultiSurfaceSuspensionPolicy(WHEELS, .12, .12)
POLICY = IndependentSuspensionDynamicsPolicy(
    CONTACT, mass_kg=1200., pitch_inertia_kg_m2=1800.,
    roll_inertia_kg_m2=600., spring_stiffness_npm=30000.,
    damper_nspm=2000., angular_damping_nmsprad=150.,
    integration_substeps=8,
)


def patch(name, z, length=4., y=0., angle=0.):
    forward = Vec3(0., sin(angle), cos(angle))
    normal = Vec3(0., cos(angle), -sin(angle))
    return PlanarSurfacePatch(name, Vec3(0., y, z), forward, normal, 4., length)


def car(x=0., z=0., height=.3, orientation=None):
    return VehicleState(Vec3(x, height, z), Vec3(0., 0., 0.),
                        orientation or Quaternion.identity(),
                        Vec3(0., 0., 0.), True)


def advance(previous, *, target=None, initial=None, roads=None, policy=POLICY, dt=.04):
    initial = initial or SuspensionDynamicsState(previous.position.y)
    target = target or previous
    roads = roads if roads is not None else (patch("road", 0., length=8.),)
    return integrate_independent_suspension(
        previous, target, initial, policy, BODY, roads,
        dt=dt, gravity_mps2=9.81,
    )


def test_flat_rest_equilibrium_no_spontaneous_heave_pitch_or_roll():
    p = car()
    result = advance(p)
    assert result.vehicle.position.y == pytest.approx(.3, abs=1e-12)
    assert result.dynamics.heave_velocity_mps == pytest.approx(0., abs=1e-12)
    assert result.dynamics.pitch_x_rad == pytest.approx(0., abs=1e-12)
    assert result.dynamics.roll_z_rad == pytest.approx(0., abs=1e-12)
    assert len(result.poses) == 9
    assert len(result.force_samples) == 8
    for wheel in result.force_samples[0]:
        assert wheel.upward_force_n == pytest.approx(1200 * 9.81 / 4)


def test_front_on_ramp_rear_on_flat_causes_independent_pitch_response():
    flat = patch("flat", -2.)
    angle = .08
    ramp = patch("ramp", 2 * cos(angle), y=2 * sin(angle), angle=angle)
    result = advance(car(z=.2), roads=(flat, ramp))
    loads = result.force_samples[0]
    assert [f.contact.patch_id for f in loads] == ["ramp", "ramp", "flat", "flat"]
    assert loads[0].upward_force_n > loads[2].upward_force_n
    assert result.dynamics.pitch_x_rad < 0.  # front rises under uphill loading
    assert result.dynamics.height_m > .3


def test_left_right_height_asymmetry_produces_roll_torque():
    angle = .06
    raised_right = PlanarSurfacePatch(
        "banked", Vec3(0., 0., 0.), Vec3(0., 0., 1.),
        Vec3(-sin(angle), cos(angle), 0.), 4., 8.,
    )
    result = advance(car(), roads=(raised_right,))
    samples = result.force_samples[0]
    assert samples[1].upward_force_n > samples[0].upward_force_n
    assert result.dynamics.roll_z_rad > 0.


def test_damping_reduces_upward_force_when_chassis_moving_up():
    p = car()
    baseline = advance(p)
    rising = advance(p, initial=SuspensionDynamicsState(.3, heave_velocity_mps=.2))
    assert rising.force_samples[0][0].upward_force_n < baseline.force_samples[0][0].upward_force_n
    assert rising.dynamics.heave_velocity_mps < .2


def test_dynamics_repeat_exactly_with_same_inputs():
    p = car()
    a = advance(p)
    b = advance(p)
    assert a == b


def test_substep_path_moves_horizontal_position_prescribed_by_integrator():
    p = car(z=-.2)
    result = advance(p, target=car(z=.2))
    assert result.vehicle.position.z == pytest.approx(.2)
    assert result.poses[4].position.z == pytest.approx(0.)
    assert all(p.grounded for p in result.poses)


def test_loss_of_wheel_support_is_not_implicitly_airborne_pass():
    p = car()
    with pytest.raises(SuspensionResolutionError, match="support|uncertified"):
        advance(p, roads=(patch("tiny", 0., length=.5),))


def test_chassis_starts_too_low_fails_closed():
    p = car(height=.1)
    with pytest.raises(SuspensionResolutionError, match="support|uncertified"):
        advance(p)


def test_discontinuous_stacked_road_atlas_rejected():
    p = car()
    with pytest.raises(SuspensionResolutionError, match="atlas"):
        advance(p, roads=(patch("low", 0., length=8.), patch("high", 0., length=8., y=2.)))


def test_missing_persistent_state_cannot_be_recreated_from_any_height():
    p = car()
    with pytest.raises(SuspensionResolutionError, match="stored chassis height"):
        advance(p, initial=SuspensionDynamicsState(.5))


def test_state_attitude_must_be_in_sync_with_vehicle_pose():
    p = car()
    with pytest.raises(SuspensionResolutionError, match="attitude disagrees"):
        advance(p, initial=SuspensionDynamicsState(.3, pitch_x_rad=.1))


def test_airborne_candidate_not_silently_accepted():
    p = car()
    with pytest.raises(SuspensionResolutionError, match="airborne"):
        advance(p, target=replace(p, grounded=False))


def test_tilt_budget_is_fail_closed():
    p = car()
    severe = replace(POLICY, max_abs_tilt_rad=.001)
    angle = .08
    ramp = patch("ramp", 2 * cos(angle), y=2 * sin(angle), angle=angle)
    with pytest.raises(SuspensionResolutionError, match="tilt"):
        advance(car(z=.2), roads=(patch("flat", -2.), ramp), policy=severe, dt=.15)


@pytest.mark.parametrize("change", [
    {"mass_kg": 0.}, {"spring_stiffness_npm": float("inf")},
    {"damper_nspm": -1.}, {"pitch_inertia_kg_m2": True},
    {"integration_substeps": 0}, {"integration_substeps": 129},
    {"max_abs_tilt_rad": .8},
])
def test_invalid_research_physical_parameters_rejected(change):
    with pytest.raises(ValueError):
        replace(POLICY, **change)


def test_world_opt_in_writes_persistent_suspension_and_telemetry():
    from src.simulation.interactions.state import InteractionState
    from src.simulation.rules.scoring import ScoreState
    from src.simulation.track.model import TrackWorld
    from src.simulation.track.state import TrackState
    from src.simulation.vehicle import VehicleControl, VehicleParameters
    from src.simulation.world import WorldState, step_world

    p = car()
    world = WorldState(p, TrackState(frozenset()),
                       InteractionState(frozenset(), frozenset()), ScoreState(),
                       suspension=SuspensionDynamicsState(.3))
    params = VehicleParameters(2.5, .3, .6, 0., 20., 0., 0., 0., 9.81)
    args = (TrackWorld(surfaces=(patch("first", -2.), patch("second", 2.))),
            world, VehicleControl(0., 0., 0.), params, .02, .5)
    result = step_world(*args, body_geometry=BODY, independent_suspension_policy=POLICY)
    assert result.state.suspension is not None
    assert result.suspension_telemetry is not None
    assert len(result.suspension_telemetry.force_samples) == POLICY.integration_substeps
    assert result.state.vehicle.position.y == pytest.approx(.3)


def test_world_refuses_suspension_without_explicit_initial_state_or_geometry():
    from src.simulation.interactions.state import InteractionState
    from src.simulation.rules.scoring import ScoreState
    from src.simulation.track.model import TrackWorld
    from src.simulation.track.state import TrackState
    from src.simulation.vehicle import VehicleControl, VehicleParameters
    from src.simulation.world import WorldState, step_world

    p = car()
    state = WorldState(p, TrackState(frozenset()),
                       InteractionState(frozenset(), frozenset()), ScoreState())
    params = VehicleParameters(2.5, .3, .6, 0., 20., 0., 0., 0., 9.81)
    args = (TrackWorld(surfaces=(patch("road", 0., length=8.),)),
            state, VehicleControl(0., 0., 0.), params, .02, .5)
    with pytest.raises(ValueError, match="requires explicit"):
        step_world(*args, body_geometry=BODY, independent_suspension_policy=POLICY)
    with pytest.raises(ValueError, match="requires explicit"):
        step_world(*args, independent_suspension_policy=POLICY)


def test_world_sweeps_body_obstacles_across_all_suspension_substeps():
    from src.simulation.entities.obstacle import Obstacle
    from src.simulation.interactions.events import ObstacleHit
    from src.simulation.interactions.state import InteractionState
    from src.simulation.rules.scoring import ScoreState
    from src.simulation.track.model import TrackWorld
    from src.simulation.track.state import TrackState
    from src.simulation.track.volume import OrientedBoxVolume
    from src.simulation.vehicle import VehicleControl, VehicleParameters
    from src.simulation.world import WorldState, step_world

    start = replace(car(z=-2.), velocity=Vec3(0., 0., 25.))
    barrier = Obstacle("barrier", OrientedBoxVolume(
        Vec3(0., .3, 0.), Vec3(.5, .5, .05), Quaternion.identity(),
    ))
    world = WorldState(
        start, TrackState(frozenset()),
        InteractionState(frozenset(), frozenset()), ScoreState(),
        suspension=SuspensionDynamicsState(.3),
    )
    params = VehicleParameters(2.5, .3, .6, 0., 20., 0., 0., 50., 9.81)
    result = step_world(
        TrackWorld(surfaces=(patch("road", 0., length=50.),), obstacles=(barrier,)),
        world, VehicleControl(0., 0., 0.), params, .2, .5,
        body_geometry=BODY, independent_suspension_policy=POLICY,
    )
    assert result.events == (ObstacleHit("barrier"),)
    assert result.state.vehicle.position == start.position
    assert result.state.suspension == world.suspension
    assert result.suspension_telemetry is None


def test_world_suspension_state_continues_across_sequential_steps():
    from src.simulation.interactions.state import InteractionState
    from src.simulation.rules.scoring import ScoreState
    from src.simulation.track.model import TrackWorld
    from src.simulation.track.state import TrackState
    from src.simulation.vehicle import VehicleControl, VehicleParameters
    from src.simulation.world import WorldState, step_world

    p = car()
    state = WorldState(p, TrackState(frozenset()),
                       InteractionState(frozenset(), frozenset()), ScoreState(),
                       suspension=SuspensionDynamicsState(.3))
    params = VehicleParameters(2.5, .3, .6, 0., 20., 0., 0., 0., 9.81)
    args = (TrackWorld(surfaces=(patch("road", 0., length=8.),)),
            VehicleControl(0., 0., 0.), params, .02, .5)
    first = step_world(args[0], state, *args[1:],
                       body_geometry=BODY, independent_suspension_policy=POLICY)
    second = step_world(args[0], first.state, *args[1:],
                        body_geometry=BODY, independent_suspension_policy=POLICY)
    assert second.state.suspension is not None
    assert second.state.vehicle.position.y == pytest.approx(.3)
    assert second.suspension_telemetry is not None


def test_no_hit_obstacle_does_not_break_full_body_sweep():
    from src.simulation.entities.obstacle import Obstacle
    from src.simulation.interactions.state import InteractionState
    from src.simulation.rules.scoring import ScoreState
    from src.simulation.track.model import TrackWorld
    from src.simulation.track.state import TrackState
    from src.simulation.track.volume import OrientedBoxVolume
    from src.simulation.vehicle import VehicleControl, VehicleParameters
    from src.simulation.world import WorldState, step_world

    p = car()
    obstacle = Obstacle("far", OrientedBoxVolume(
        Vec3(100., 2., 100.), Vec3(1., 1., 1.), Quaternion.identity(),
    ))
    state = WorldState(p, TrackState(frozenset()),
                       InteractionState(frozenset(), frozenset()), ScoreState(),
                       suspension=SuspensionDynamicsState(.3))
    parameters = VehicleParameters(2.5, .3, .6, 0., 20., 0., 0., 0., 9.81)
    result = step_world(
        TrackWorld(surfaces=(patch("road", 0., length=8.),), obstacles=(obstacle,)),
        state, VehicleControl(0., 0., 0.), parameters, .02, .5,
        body_geometry=BODY, independent_suspension_policy=POLICY,
    )
    assert result.events == ()
    assert result.state.vehicle.position == p.position
