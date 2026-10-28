"""New prospective continuous body engine; no original PPO runs are replayed."""
from dataclasses import replace
from math import pi

import pytest

from src.sandbox.body_collision import (
    VehicleBodyGeometry, obb_separation_lower_bound, sweep_body_against_obstacle,
)
from src.sandbox.entities.obstacle import Obstacle
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.geometry.vector import Vec3
from src.sandbox.interactions.events import ObstacleHit
from src.sandbox.interactions.state import InteractionState
from src.sandbox.rules.scoring import ScoreState
from src.sandbox.track.model import TrackWorld
from src.sandbox.track.patch import PlanarSurfacePatch
from src.sandbox.track.state import TrackState
from src.sandbox.track.volume import OrientedBoxVolume
from src.sandbox.vehicle import VehicleControl, VehicleParameters, VehicleState
from src.sandbox.world import WorldState, step_world


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
    assert obb_separation_lower_bound(beam.box_at(Vec3(0.,0.,0.),left),obstacle)>0.
    assert obb_separation_lower_bound(beam.box_at(Vec3(0.,0.,0.),right),obstacle)>0.
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
    import src.sandbox.world as world_module
    from src.sandbox.body_collision import SweptCollisionResult

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
