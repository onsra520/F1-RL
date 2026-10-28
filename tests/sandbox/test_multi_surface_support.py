"""Synthetic independent wheel support regressions; HELD-OUT tracks never loaded."""
from dataclasses import replace
from math import pi, sin, cos

import pytest

from src.sandbox.body_collision import VehicleBodyGeometry
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.geometry.vector import Vec3
from src.sandbox.multi_surface_support import (
    MultiSurfaceSuspensionPolicy, certify_multisurface_support,
)
from src.sandbox.ramp_support import WheelContactPoints
from src.sandbox.track.patch import PlanarSurfacePatch
from src.sandbox.vehicle import VehicleState

BODY = VehicleBodyGeometry(1., .4, 2., Vec3(0., .1, 0.))
WHEELS = WheelContactPoints(Vec3(-.4, -.3, .8), Vec3(.4, -.3, .8),
                            Vec3(-.4, -.3, -.8), Vec3(.4, -.3, -.8))
POLICY = MultiSurfaceSuspensionPolicy(WHEELS, .12, .12)
UP = Vec3(0., 1., 0.)
FWD = Vec3(0., 0., 1.)


def patch(name, z, length=4., y=0., angle=0.):
    forward = Vec3(0., sin(angle), cos(angle))
    normal = Vec3(0., cos(angle), -sin(angle))
    return PlanarSurfacePatch(name, Vec3(0., y, z), forward, normal, 4., length)


def car(z, y=.3, rotation=None):
    return VehicleState(Vec3(0., y, z), Vec3(0., 0., 0.),
                        rotation or Quaternion.identity(), Vec3(0., 0., 0.), True)


def report(start, end, surfaces, policy=POLICY):
    return certify_multisurface_support(start, end, BODY, policy, surfaces)


def test_independent_contacts_switch_across_contiguous_flat_seam():
    a, b = patch("z-last", -2.), patch("a-first", 2.)
    result = report(car(-1.), car(1.), (b, a))
    assert result.status == "clear", result
    assert result.valid and result.checks >= 3
    # Resolved contacts are independent, not a single centre-plane sample.
    midpoint = report(car(0.), car(0.), (a, b))
    assert midpoint.valid
    assert [c.patch_id for c in midpoint.contacts[:2]] == ["a-first"] * 2
    assert [c.patch_id for c in midpoint.contacts[2:]] == ["z-last"] * 2


def test_patch_input_order_does_not_change_contact_selection():
    a, b = patch("z-last", -2.), patch("a-first", 2.)
    first = report(car(0.), car(0.), (a, b))
    second = report(car(0.), car(0.), (b, a))
    assert first == second


def test_front_tires_ramp_rear_tires_flat_with_explicit_travel():
    angle = .08
    flat = patch("flat", -2.)
    # Begin the inclined patch at xz=(0,0), with elevation 0 at the seam.
    ramp = patch("ramp", 2 * cos(angle), y=2 * sin(angle), angle=angle)
    result = report(car(.2), car(.2), (flat, ramp))
    assert result.valid, result
    assert [c.patch_id for c in result.contacts] == ["ramp", "ramp", "flat", "flat"]
    assert result.contacts[0].deflection_m > 0.
    assert result.contacts[2].deflection_m == pytest.approx(0.)


def test_large_height_difference_rejected_even_when_two_decks_overlap():
    bottom = patch("bottom", 0., y=0.)
    top = patch("top", 0., y=3.)
    result = report(car(0.), car(0.), (bottom, top))
    assert result.status == "unsupported"


def test_discontinuous_step_at_touching_seam_fails_closed():
    bottom = patch("bottom", -2.)
    step = patch("step", 2., y=.15)
    assert report(car(0.), car(0.), (bottom, step)).status == "unsupported"


def test_gap_beneath_chassis_not_hidden_by_independent_tire_support():
    a = patch("before", -2.08)
    b = patch("after", 2.08)
    result = report(car(0.), car(0.), (a, b))
    assert result.status == "outside"
    assert not result.valid


def test_unsupported_wheel_deflection_returns_outside():
    road = patch("road", 0., length=8.)
    lifted = car(0., y=.6)
    assert report(lifted, lifted, (road,)).status == "outside"


def test_pitch_causes_independent_front_rear_deflection_failure():
    road = patch("road", 0., length=8.)
    pitched = car(0., rotation=Quaternion.from_axis_angle(Vec3(1., 0., 0.), .4))
    assert report(pitched, pitched, (road,)).status == "outside"


def test_body_penetration_rejected_even_if_tires_are_supported():
    road = patch("road", 0., length=8.)
    penetrating = VehicleBodyGeometry(1., 1.2, 2., Vec3(0., 0., 0.))
    result = certify_multisurface_support(car(0.), car(0.), penetrating, POLICY, (road,))
    assert result.status == "outside"


def test_outside_at_midpoint_cannot_be_mistaken_for_clear():
    # Motion endpoints are geometrically valid but the body sweeps off road.
    road = patch("road", 0., length=8.)
    first = car(0., rotation=Quaternion.identity())
    last = car(0., rotation=Quaternion.from_axis_angle(UP, pi))
    narrow = PlanarSurfacePatch("narrow", road.origin, FWD, UP, 1.6, 8.)
    result = report(first, last, (narrow,))
    assert result.status in ("outside", "uncertain")
    assert not result.valid


def test_airborne_not_certified_as_road_contact():
    road = patch("road", 0., length=8.)
    a = car(0.)
    assert report(replace(a, grounded=False), a, (road,)).status == "unsupported"


@pytest.mark.parametrize("key,value",[
    ("max_bump_m", 0.), ("max_droop_m", float("inf")),
    ("seam_height_tolerance_m", True), ("max_subdivisions", -1),
])
def test_invalid_policy_rejected(key, value):
    kwargs = dict(wheels=WHEELS, max_bump_m=.12, max_droop_m=.12)
    kwargs[key] = value
    with pytest.raises(ValueError):
        MultiSurfaceSuspensionPolicy(**kwargs)


def test_world_multi_surface_policy_is_explicit_and_mutually_exclusive():
    from src.sandbox.interactions.state import InteractionState
    from src.sandbox.ramp_support import PlanarRampPolicy
    from src.sandbox.rules.scoring import ScoreState
    from src.sandbox.track.model import TrackWorld
    from src.sandbox.track.state import TrackState
    from src.sandbox.vehicle import VehicleControl, VehicleParameters
    from src.sandbox.world import WorldState, step_world

    road = TrackWorld(surfaces=(patch("before", -2.), patch("after", 2.)))
    original = car(0.)
    world = WorldState(original, TrackState(frozenset()),
                       InteractionState(frozenset(), frozenset()), ScoreState())
    params = VehicleParameters(2.5, .3, .6, 0., 20., 0., 0., 0., 9.81)
    args = (road, world, VehicleControl(0., 0., 0.), params, .02, .5)
    with pytest.raises(ValueError, match="multi_surface_policy requires"):
        step_world(*args, multi_surface_policy=POLICY)
    with pytest.raises(ValueError, match="mutually exclusive"):
        step_world(*args, body_geometry=BODY, multi_surface_policy=POLICY,
                   ramp_support_policy=PlanarRampPolicy(WHEELS, .02))
    result = step_world(*args, body_geometry=BODY, multi_surface_policy=POLICY)
    assert result.state.vehicle.grounded
    assert result.events == ()
