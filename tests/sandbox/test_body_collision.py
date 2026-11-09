"""New prospective continuous body engine; no original PPO runs are replayed."""
from dataclasses import replace
from decimal import Decimal
from math import nextafter, pi
from random import Random

import pytest

from f1rl.sandbox.body_collision import (
    VehicleBodyGeometry,
    obb_separation_at_most,
    obb_separation_lower_bound,
    sweep_body_against_obstacle,
)
from f1rl.sandbox.entities.obstacle import Obstacle
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.interactions.events import ObstacleHit
from f1rl.sandbox.interactions.state import InteractionState
from f1rl.sandbox.rules.scoring import ScoreState
from f1rl.sandbox.track.model import TrackWorld
from f1rl.sandbox.track.patch import PlanarSurfacePatch
from f1rl.sandbox.track.state import TrackState
from f1rl.sandbox.track.volume import OrientedBoxVolume
from f1rl.sandbox.vehicle import VehicleControl, VehicleParameters, VehicleState
from f1rl.sandbox.world import WorldState, step_world
from tests.sandbox.sat_reference import (
    obb_separation_exact,
    obb_separation_vertices_exact,
)


IDENTITY = Quaternion.identity()
BODY = VehicleBodyGeometry(0.8, 0.8, 1.2, Vec3(0., 0., 0.))


def state(x: float = 0., y: float = 0., z: float = 0., q: Quaternion = IDENTITY) -> VehicleState:
    return VehicleState(
        Vec3(x, y, z), Vec3(0., 0., 0.), q,
        Vec3(0., 0., 0.), True,
    )


def volume(x=0., y=0., z=0., hx=0.5, hy=0.5, hz=0.5, q=IDENTITY) -> OrientedBoxVolume:
    return OrientedBoxVolume(Vec3(x,y,z),Vec3(hx,hy,hz),q)


def test_sat_separation_and_touching() -> None:
    candidate = BODY.box_at(Vec3(3.,0.,0.),IDENTITY)
    obstacle = volume()
    assert obb_separation_lower_bound(candidate,obstacle)==pytest.approx(3.-.4-.5)
    assert obb_separation_lower_bound(BODY.box_at(Vec3(.9,0.,0.),IDENTITY),obstacle)==pytest.approx(0.,abs=1e-10)
    assert obb_separation_lower_bound(BODY.box_at(Vec3(0.,0.,0.),IDENTITY),obstacle)<0.


def test_near_parallel_cross_axis_uses_the_independent_vertex_oracle():
    first = OrientedBoxVolume(
        Vec3(0., 0., 0.),
        Vec3(.9, .6, 2.),
        Quaternion(
            .7995053347671702,
            -.28612311668290946,
            -.48553691075811284,
            .20779482686044343,
        ),
    )
    second = OrientedBoxVolume(
        Vec3(6.4, .05110225107717026, -.288274099663574),
        Vec3(.5, .7, 1.6),
        Quaternion(
            .7995053347671408,
            -.286123116682931,
            -.4855369107581395,
            .20779482686046497,
        ),
    )
    threshold = 2.4
    oracle_gap = obb_separation_vertices_exact(first, second)

    assert float(oracle_gap) == pytest.approx(2.5312532935515336, abs=1e-12)
    assert not obb_separation_at_most(first, second, threshold), (
        f"expected clear from oracle_gap={oracle_gap}; "
        f"float_gap={obb_separation_lower_bound(first, second)}"
    )


def test_near_parallel_sat_matches_vertex_oracle_across_seeded_thresholds():
    first = OrientedBoxVolume(
        Vec3(0., 0., 0.), Vec3(.9, .6, 2.),
        Quaternion(.7995053347671702, -.28612311668290946,
                   -.48553691075811284, .20779482686044343),
    )
    base_orientation = Quaternion(
        .7995053347671408, -.286123116682931,
        -.4855369107581395, .20779482686046497,
    )
    base_center = Vec3(6.4, .05110225107717026, -.288274099663574)
    random = Random(0x4E415253)
    comparisons = 0
    positive = 0
    negative = 0

    for seed in (3, 17, 412, 20261110):
        random.seed(seed)
        axis = Vec3(random.uniform(-1., 1.), random.uniform(-1., 1.),
                    random.uniform(-1., 1.)).normalized()
        epsilon = random.uniform(1e-14, 2e-13)
        orientation = Quaternion.from_axis_angle(axis, epsilon) * base_orientation
        center = base_center + Vec3(
            random.uniform(-.01, .01),
            random.uniform(-.01, .01),
            random.uniform(-.01, .01),
        )
        second = OrientedBoxVolume(center, Vec3(.5, .7, 1.6), orientation)
        oracle_gap = obb_separation_vertices_exact(first, second)
        rounded_gap = float(oracle_gap)
        thresholds = tuple(sorted({
            0., 1e-9, 2.4, 2.6, 3.,
            max(0., nextafter(rounded_gap, float("-inf"))),
            rounded_gap,
            nextafter(rounded_gap, float("inf")),
        }))

        for threshold in thresholds:
            expected = oracle_gap <= Decimal.from_float(threshold)
            actual = obb_separation_at_most(first, second, threshold)
            repeated = obb_separation_at_most(first, second, threshold)
            assert actual is expected, (
                f"seed={seed} threshold={threshold} oracle_gap={oracle_gap}"
            )
            assert repeated is actual, f"non-deterministic decision for seed={seed}"
            positive += int(expected)
            negative += int(not expected)
            comparisons += 1

    assert comparisons == 32
    assert positive > 0 and negative > 0


def test_shared_sat_comparator_and_sweep_resolve_the_same_threshold_boundary():
    road = PlanarSurfacePatch(
        "road", Vec3(60.0, -0.8884474173192283, 0.5604337803032049),
        Vec3(0.2810253020973806, -0.029913293028186447, 0.9592340561517226),
        Vec3(-0.28737537016871983, 0.9510277254621118, 0.11384929522291205),
        12.0, 8.0,
    )
    orientation = Quaternion.from_basis(road.right, road.normal, road.forward)
    obstacle = OrientedBoxVolume(
        road.origin * 0.1, Vec3(0.6, 1e-10, 0.4), orientation,
    )
    body = VehicleBodyGeometry(0.18, 0.12, 0.4, Vec3(0., 0., 0.))
    x, z = 59.82757477732402 * 0.1, 0.6287433576646507 * 0.1
    y = -0.31783078013990584 * 0.1
    for _ in range(27):
        y = nextafter(y, float("inf"))
    within_position = Vec3(x, y, z)
    within_threshold = body.box_at(within_position, orientation)
    y = nextafter(y, float("inf"))
    outside_position = Vec3(x, y, z)
    outside_threshold = body.box_at(outside_position, orientation)
    threshold = 1e-10

    assert obb_separation_exact(within_threshold, obstacle) <= Decimal.from_float(threshold)
    assert obb_separation_lower_bound(within_threshold, obstacle) <= threshold
    assert obb_separation_at_most(within_threshold, obstacle, threshold) is True
    assert obb_separation_exact(outside_threshold, obstacle) > Decimal.from_float(threshold)
    assert obb_separation_lower_bound(outside_threshold, obstacle) <= threshold
    assert obb_separation_at_most(outside_threshold, obstacle, threshold) is False

    contact = state(within_position.x, within_position.y, within_position.z, orientation)
    clear = state(outside_position.x, outside_position.y, outside_position.z, orientation)
    assert sweep_body_against_obstacle(contact, contact, body, obstacle).contact
    clear_result = sweep_body_against_obstacle(clear, clear, body, obstacle)
    assert not clear_result.contact and not clear_result.uncertain


def test_shared_sat_comparator_rejects_invalid_geometry_and_thresholds():
    invalid_box = OrientedBoxVolume(None, Vec3(1., 1., 1.), IDENTITY)
    with pytest.raises(ValueError, match="finite OBB"):
        obb_separation_at_most(invalid_box, volume(), 0.)
    for threshold in (float("nan"), float("inf"), -1., True):
        with pytest.raises(ValueError, match="SAT threshold"):
            obb_separation_at_most(volume(), volume(), threshold)


def test_fast_translation_does_not_tunnel_through_thin_box() -> None:
    obstacle=volume(hx=.05)
    assert not obstacle.contains(Vec3(-5,0,0))
    assert not obstacle.contains(Vec3(5,0,0))
    report=sweep_body_against_obstacle(state(-5),state(5),BODY,obstacle)
    assert report.blocked and report.contact and not report.uncertain


def test_far_parallel_motion_is_proved_clear() -> None:
    report=sweep_body_against_obstacle(state(-5,z=5),state(5,z=5),BODY,volume())
    assert not report.blocked
    assert not report.uncertain


def test_rotated_obstacle_and_slanted_intersection() -> None:
    q=Quaternion.from_axis_angle(Vec3(0.,1.,0.),pi/4.)
    report=sweep_body_against_obstacle(
        state(z=-5),state(z=5),BODY,
        volume(hx=.2,hy=1.,hz=1.,q=q),
    )
    assert report.contact and report.blocked


def test_body_center_offset_is_local_and_collides() -> None:
    moved=VehicleBodyGeometry(1.,1.,1.,Vec3(-1.5,0.,0.))
    report=sweep_body_against_obstacle(
        state(x=2.),state(x=2.),moved,volume(),
    )
    assert report.contact and report.blocked
    assert not volume().contains(Vec3(2.,0.,0.))


def test_rotation_only_body_sweep_hits_between_clear_endpoints() -> None:
    beam=VehicleBodyGeometry(.3,.4,4.,Vec3(0.,0.,0.))
    left=Quaternion.from_axis_angle(Vec3(0.,1.,0.),-pi/2.)
    right=Quaternion.from_axis_angle(Vec3(0.,1.,0.),pi/2.)
    obstacle=volume(z=1.7,hx=.18,hy=.3,hz=.18)
    assert not obb_separation_at_most(
        beam.box_at(Vec3(0.,0.,0.),left),obstacle,0.,
    )
    assert not obb_separation_at_most(
        beam.box_at(Vec3(0.,0.,0.),right),obstacle,0.,
    )
    report=sweep_body_against_obstacle(state(q=left),state(q=right),beam,obstacle)
    assert report.contact and report.blocked


def test_exhausted_resolution_fails_closed_without_false_contact_claim() -> None:
    thin=VehicleBodyGeometry(.6,.6,.6,Vec3(0.,0.,0.))
    obstacle=volume()
    low=sweep_body_against_obstacle(
        state(x=-4,z=1.1),state(x=4,z=1.1),
        thin,obstacle,max_subdivisions=0,
    )
    assert low.blocked and low.uncertain and not low.contact
    high=sweep_body_against_obstacle(
        state(x=-4,z=1.1),state(x=4,z=1.1),
        thin,obstacle,max_subdivisions=12,
    )
    assert not high.blocked


@pytest.mark.parametrize("bad", [
    (-1.,1.,1.),
    (1.,0.,1.),
    (1.,1.,float("inf")),
    (True,1.,1.),
])
def test_unsafe_dimensions_are_rejected(bad) -> None:
    with pytest.raises(ValueError):
        VehicleBodyGeometry(*bad,Vec3(0.,0.,0.))


@pytest.mark.parametrize("limit",[-1,21,True,1.5])
def test_invalid_sweep_depth_rejected(limit) -> None:
    with pytest.raises(ValueError):
        sweep_body_against_obstacle(state(-1),state(1),BODY,volume(),max_subdivisions=limit)


def test_world_step_legacy_kept_and_body_opt_in_blocks_tunnelling() -> None:
    road=PlanarSurfacePatch(
        "road",Vec3(0.,0.,0.),Vec3(0.,0.,1.),
        Vec3(0.,1.,0.),20.,100.,
    )
    barrier=Obstacle("barrier",volume(y=.3,z=0.,hz=.12))
    track=TrackWorld(surfaces=(road,),obstacles=(barrier,))
    previous=VehicleState(
        Vec3(0.,.3,-2.),Vec3(0.,0.,25.),
        IDENTITY,Vec3(0.,0.,0.),True,
    )
    world=WorldState(
        previous,TrackState(frozenset()),
        InteractionState(frozenset(),frozenset()),ScoreState(),
    )
    params=VehicleParameters(2.5,.3,.6,0.,20.,0.,0.,50.,9.81)
    inputs=(track,world,VehicleControl(0.,0.,0.),params,.2,.5)
    legacy=step_world(*inputs)
    aware=step_world(*inputs,body_geometry=VehicleBodyGeometry(1.,1.,2.,Vec3(0.,0.,0.)))
    assert legacy.state.vehicle.position.z>0.
    assert legacy.events==()
    assert aware.state.vehicle.position==previous.position
    assert aware.events==(ObstacleHit("barrier"),)
    assert aware.state.vehicle.velocity==Vec3(0.,0.,0.)


def test_world_rejects_invalid_optin_type() -> None:
    road=PlanarSurfacePatch(
        "road",Vec3(0.,0.,0.),Vec3(0.,0.,1.),
        Vec3(0.,1.,0.),20.,100.,
    )
    initial=state()
    world=WorldState(initial,TrackState(frozenset()),
        InteractionState(frozenset(),frozenset()),ScoreState())
    params=VehicleParameters(2.5,.3,.6,0.,20.,0.,0.,50.,9.81)
    with pytest.raises(ValueError,match="body_geometry"):
        step_world(TrackWorld(surfaces=(road,)),world,
            VehicleControl(0.,0.,0.),params,.1,.5,body_geometry="car")


def test_unresolved_world_sweep_invalidates_rollout_without_fake_hit(monkeypatch) -> None:
    import f1rl.sandbox.world as world_module
    from f1rl.sandbox.body_collision import SweptCollisionResult

    road=PlanarSurfacePatch(
        "road",Vec3(0.,0.,0.),Vec3(0.,0.,1.),
        Vec3(0.,1.,0.),20.,100.,
    )
    track=TrackWorld(surfaces=(road,),obstacles=(Obstacle("uncertain",volume(z=20.)),))
    previous=state(y=.3)
    world=WorldState(previous,TrackState(frozenset()),
        InteractionState(frozenset(),frozenset()),ScoreState())
    params=VehicleParameters(2.5,.3,.6,0.,20.,0.,0.,50.,9.81)
    monkeypatch.setattr(
        world_module,"sweep_body_against_obstacle",
        lambda *args,**kwargs: SweptCollisionResult(False,True,1),
    )
    with pytest.raises(RuntimeError,match="rollout is invalid"):
        step_world(track,world,VehicleControl(0.,0.,0.),params,.1,.5,body_geometry=BODY)
    # The original WorldState remains immutable and no fictitious hit is emitted.
    assert world.vehicle==previous
