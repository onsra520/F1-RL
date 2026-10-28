"""Prospective whole-body continuous road-union validity; TRAIN-neutral geometry tests.

No HELD-OUT asset content, checkpoint, PPO policy or historical manifest read.
"""
from math import pi

import pytest

from src.sandbox.body_collision import VehicleBodyGeometry
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.geometry.vector import Vec3
from src.sandbox.interactions.state import InteractionState
from src.sandbox.road_footprint import (
    body_xz_polygon, swept_body_road_union,
)
from src.sandbox.rules.scoring import ScoreState
from src.sandbox.track.model import TrackWorld
from src.sandbox.track.patch import PlanarSurfacePatch
from src.sandbox.track.state import TrackState
from src.sandbox.vehicle import VehicleControl, VehicleParameters, VehicleState
from src.sandbox.world import WorldState, step_world


IDENTITY=Quaternion.identity()
BODY=VehicleBodyGeometry(1.,1.,2.,Vec3(0.,0.,0.))
RIDE=.3


def state(*,x:float=0.,y:float=RIDE,z:float=0.,
          yaw:float=0.,rotation:Quaternion|None=None,
          grounded:bool=True)->VehicleState:
    q=rotation if rotation is not None else Quaternion.from_axis_angle(Vec3(0.,1.,0.),yaw)
    return VehicleState(Vec3(x,y,z),Vec3(0.,0.,0.),
                        q,Vec3(0.,0.,0.),grounded)


def north_road(*,origin:Vec3=Vec3(0.,0.,0.),
               width:float=4.,length:float=10.,id:str="north")->PlanarSurfacePatch:
    return PlanarSurfacePatch(id,origin,Vec3(0.,0.,1.),Vec3(0.,1.,0.),width,length)


def east_road(*,origin:Vec3=Vec3(5.,0.,0.),
              width:float=4.,length:float=6.,id:str="east")->PlanarSurfacePatch:
    return PlanarSurfacePatch(id,origin,Vec3(1.,0.,0.),Vec3(0.,1.,0.),width,length)


def run(start:VehicleState,end:VehicleState,patches:tuple[PlanarSurfacePatch,...],
        *,body:VehicleBodyGeometry=BODY,max_subdivisions:int=12):
    return swept_body_road_union(
        start,end,body,patches,ride_height_m=RIDE,
        max_subdivisions=max_subdivisions,
    )


def test_whole_body_can_travel_through_flat_road() -> None:
    result=run(state(z=-2),state(z=2),(north_road(),))
    assert result.valid and result.status=="clear"
    assert result.checks>2


def test_body_overhang_detected_when_center_is_inside() -> None:
    result=run(state(x=1.8),state(x=1.8),(north_road(),))
    assert result.status=="outside" and not result.valid
    # Centre x=1.8 fits width/2=2, but body extends to 2.3.


def test_full_union_accepts_body_straddling_two_road_patches() -> None:
    # North road spans x[-2,2]; east road spans x[2,8].
    # Body spans x[1,3], z[-1,1]: neither patch alone covers it.
    roads=(north_road(length=10.),east_road())
    assert run(state(x=2.),state(x=2.),roads).valid
    assert run(state(x=2.),state(x=2.),(roads[0],)).status=="outside"
    assert run(state(x=2.),state(x=2.),(roads[1],)).status=="outside"


def test_gap_between_patches_is_detected_mid_timestep() -> None:
    roads=(
        north_road(origin=Vec3(0.,0.,-3.),length=4.,id="start"),
        north_road(origin=Vec3(0.,0.,3.),length=4.,id="end"),
    )
    narrow=VehicleBodyGeometry(.6,.8,.4,Vec3(0.,0.,0.))
    result=run(state(z=-2.),state(z=2.),roads,body=narrow)
    assert result.status=="outside"


def test_rotating_body_exits_road_between_clear_endpoints() -> None:
    # At yaw +/-90 deg the car is aligned along X; midpoint at yaw 0
    # extends along Z and violates a thin east-west road.
    narrow_east=east_road(origin=Vec3(0.,0.,0.),width=1.,length=5.)
    long_body=VehicleBodyGeometry(.3,.4,3.,Vec3(0.,0.,0.))
    result=run(
        state(yaw=-pi/2.),state(yaw=pi/2.),
        (narrow_east,),body=long_body,
    )
    assert result.status=="outside"


def test_3d_roll_increases_horizontal_footprint() -> None:
    tall=VehicleBodyGeometry(1.,4.,1.,Vec3(0.,0.,0.))
    roll=Quaternion.from_axis_angle(Vec3(0.,0.,1.),pi/4.)
    projected=body_xz_polygon(state(rotation=roll),tall)
    assert max(x for x,_ in projected)>1.
    # Thin north road width=2.0 cannot hold the projected tall body.
    result=run(state(rotation=roll),state(rotation=roll),
               (north_road(width=2.),),body=tall)
    assert result.status=="outside"


def test_unprovable_boundary_motion_is_uncertain_not_safe() -> None:
    # Entire body is exactly on x=+1 road boundary, moving in Z.
    # Conservative translational envelope crosses the edge.
    report=run(
        state(x=.5,z=-1.),state(x=.5,z=1.),
        (north_road(width=2.,length=10.),),
        max_subdivisions=0,
    )
    assert report.status=="uncertain" and not report.valid


def test_ramp_surface_requires_registered_3d_support_policy() -> None:
    slope=PlanarSurfacePatch(
        "ramp",Vec3(0.,0.,0.),
        Vec3(0.,.6,.8),Vec3(0.,.8,-.6),
        4.,10.,
    )
    report=run(state(),state(z=1.),(slope,))
    assert report.status=="unsupported"
    assert "coplanar" in report.reason


def test_different_deck_elevations_not_collapsed_to_xz() -> None:
    roads=(
        north_road(origin=Vec3(0.,0.,0.),id="lower"),
        north_road(origin=Vec3(0.,4.,0.),id="upper"),
    )
    assert run(state(),state(),roads).status=="unsupported"


def test_airborne_is_not_silently_cleared_using_xz_projection() -> None:
    report=run(state(grounded=False),state(grounded=False),(north_road(),))
    assert report.status=="unsupported"


def test_false_grounded_elevation_rejected() -> None:
    assert run(state(y=5.),state(y=5.),(north_road(),)).status=="unsupported"


def test_nothing_can_be_certified_without_road() -> None:
    assert run(state(),state(),()).status=="unsupported"


@pytest.mark.parametrize("value",[-.1,0.,float("inf"),True])
def test_ride_height_must_be_finite_positive(value) -> None:
    with pytest.raises(ValueError,match="ride_height"):
        swept_body_road_union(state(),state(),BODY,
                              (north_road(),),ride_height_m=value)


@pytest.mark.parametrize("bad",[-1,21,True,1.5])
def test_invalid_subdivision_budget_rejected(bad) -> None:
    with pytest.raises(ValueError,match="max_subdivisions"):
        run(state(),state(),(north_road(),),max_subdivisions=bad)


def test_world_legacy_accepts_center_while_fullbody_blocks_road_violation() -> None:
    road=north_road(width=2.)
    car=state()
    world=WorldState(car,TrackState(frozenset()),
        InteractionState(frozenset(),frozenset()),ScoreState())
    params=VehicleParameters(2.5,RIDE,.6,0.,20.,0.,0.,50.,9.81)
    args=(TrackWorld(surfaces=(road,)),world,VehicleControl(0.,0.,0.),
          params,.1,.5)
    legacy=step_world(*args)
    assert legacy.state.vehicle.position==car.position
    oversized=VehicleBodyGeometry(3.,1.,2.,Vec3(0.,0.,0.))
    with pytest.raises(RuntimeError,match="road validity outside"):
        step_world(*args,body_geometry=oversized)
    assert world.vehicle==car


def test_world_disallows_unreviewed_ramp_surface_in_body_mode() -> None:
    slope=PlanarSurfacePatch("ramp",Vec3(0.,0.,0.),
        Vec3(0.,.6,.8),Vec3(0.,.8,-.6),4.,10.)
    car=state()
    world=WorldState(car,TrackState(frozenset()),
        InteractionState(frozenset(),frozenset()),ScoreState())
    params=VehicleParameters(2.5,RIDE,.6,0.,20.,0.,0.,50.,9.81)
    with pytest.raises(RuntimeError,match="road validity unsupported"):
        step_world(TrackWorld(surfaces=(slope,)),world,
            VehicleControl(0.,0.,0.),params,.1,.5,
            body_geometry=BODY)
