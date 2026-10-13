"""Prospective single-planar-ramp support; synthetic geometry only, no HELD-OUT."""
from dataclasses import replace
from math import cos, pi, sin

import pytest

from f1rl.sandbox.body_collision import VehicleBodyGeometry
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.ramp_support import WheelContactPoints, certify_single_planar_ramp
from f1rl.sandbox.track.patch import PlanarSurfacePatch
from f1rl.sandbox.vehicle import VehicleState


BODY = VehicleBodyGeometry(1., 1., 2., Vec3(0., 0., 0.))
WHEELS = WheelContactPoints(
    Vec3(-.4, -.3, .8), Vec3(.4, -.3, .8),
    Vec3(-.4, -.3, -.8), Vec3(.4, -.3, -.8),
)
TOLERANCE=.02


def surface(angle:float=0.,*, width:float=4.,length:float=12.):
    forward=Vec3(0.,sin(angle),cos(angle))
    normal=Vec3(0.,cos(angle),-sin(angle))
    patch=PlanarSurfacePatch(
        "synthetic-ramp",Vec3(0.,0.,0.),forward,normal,width,length,
    )
    rotation=Quaternion.from_basis(patch.right,normal,forward)
    return patch,rotation


def car(patch:PlanarSurfacePatch,rotation:Quaternion,*,along:float=0.,
        offset:float=.3,grounded:bool=True)->VehicleState:
    return VehicleState(
        patch.origin + patch.forward*along + patch.normal*offset,
        Vec3(0.,0.,0.),rotation,Vec3(0.,0.,0.),grounded,
    )


def certify(first,last,patch,*,body=BODY,wheels=WHEELS,
            tolerance=TOLERANCE,depth=12):
    return certify_single_planar_ramp(
        first,last,body,wheels,patch,
        plane_tolerance_m=tolerance,max_subdivisions=depth,
    )


@pytest.mark.parametrize("angle",[0.,pi/12.,pi/6.])
def test_single_grounded_flat_or_sloped_patch_is_certified(angle):
    patch,q=surface(angle)
    report=certify(car(patch,q,along=-2.),car(patch,q,along=2.),patch)
    assert report.status=="clear" and report.valid
    assert report.checks>=3


def test_ramp_tire_contact_catches_false_grounded_even_with_body_inside():
    patch,q=surface(pi/6)
    # Lifting the entire pose breaks tire-ground contact without leaving
    # the projected road polygon.
    wrong=car(patch,q,offset=.5)
    result=certify(wrong,wrong,patch)
    assert result.status=="outside" and not result.valid


def test_one_tire_outside_road_but_body_still_inside():
    patch,q=surface(0.,width=2.)
    long_axles=WheelContactPoints(
        Vec3(-1.2,-.3,.8),Vec3(1.2,-.3,.8),
        Vec3(-1.2,-.3,-.8),Vec3(1.2,-.3,-.8),
    )
    result=certify(car(patch,q),car(patch,q),patch,wheels=long_axles)
    assert result.status=="outside"


def test_body_3d_projection_can_exceed_patch_with_all_wheels_supported():
    patch,q=surface(0.,width=2.)
    oversized=VehicleBodyGeometry(3.,1.,1.8,Vec3(0.,0.,0.))
    result=certify(car(patch,q),car(patch,q),patch,body=oversized)
    assert result.status=="outside"


def test_ramp_end_boundary_blocks_front_tire():
    patch,q=surface(pi/6,length=4.)
    result=certify(car(patch,q,along=1.7),car(patch,q,along=1.7),patch)
    assert result.status=="outside"


def test_mid_step_wheel_turn_leaves_road_with_clear_endpoints():
    patch,q=surface(0.,width=1.5,length=5.)
    a=car(patch,q)
    b=replace(a,orientation=Quaternion.from_axis_angle(Vec3(0.,1.,0.),pi))
    report=certify(a,b,patch)
    assert report.status=="outside" and not report.valid


def test_insufficient_resolution_is_uncertain_not_safe():
    patch,q=surface(0.,width=4.,length=12.)
    # Small yaw changes body width; coarse angular bound is insufficient.
    patch,q=surface(0.,width=1.22,length=12.)
    a=car(patch,Quaternion.from_axis_angle(Vec3(0.,1.,0.),-.1))
    b=car(patch,Quaternion.from_axis_angle(Vec3(0.,1.,0.),.1))
    result=certify(a,b,patch,depth=0)
    assert result.status=="uncertain"
    assert not result.valid
    assert certify(a,b,patch,depth=12).valid


@pytest.mark.parametrize("angle",[-pi/6.,pi/6.])
def test_grounded_false_never_proves_airborne_support(angle):
    patch,q=surface(angle)
    first=car(patch,q,grounded=False)
    result=certify(first,first,patch)
    assert result.status=="unsupported"


def test_vertical_wall_cannot_be_misidentified_as_road():
    wall=PlanarSurfacePatch("wall",Vec3(0.,0.,0.),
        Vec3(0.,1.,0.),Vec3(0.,0.,-1.),4.,5.)
    pos=car(*surface(0.))
    assert certify(pos,pos,wall).status=="unsupported"


@pytest.mark.parametrize("bad",[0.,-1.,float("inf"),True])
def test_contact_plane_tolerance_strictly_positive(bad):
    patch,q=surface(0.)
    with pytest.raises(ValueError,match="plane_tolerance_m"):
        certify(car(patch,q),car(patch,q),patch,tolerance=bad)


@pytest.mark.parametrize("bad",[-1,21,True,0.5])
def test_invalid_resolution_budget_rejected(bad):
    patch,q=surface(0.)
    with pytest.raises(ValueError,match="max_subdivisions"):
        certify(car(patch,q),car(patch,q),patch,depth=bad)


def test_wheel_specs_are_explicit_and_ordered():
    with pytest.raises(ValueError,match="four unique"):
        WheelContactPoints(Vec3(0.,0.,0.),Vec3(0.,0.,0.),
            Vec3(-1.,0.,-1.),Vec3(1.,0.,-1.))
    with pytest.raises(ValueError,match="front/rear"):
        WheelContactPoints(Vec3(-1.,0.,-1.),Vec3(1.,0.,-1.),
            Vec3(-1.,0.,1.),Vec3(1.,0.,1.))


def test_world_step_accepts_opted_in_synthetic_single_ramp():
    from f1rl.sandbox.interactions.state import InteractionState
    from f1rl.sandbox.rules.scoring import ScoreState
    from f1rl.sandbox.ramp_support import PlanarRampPolicy
    from f1rl.sandbox.track.model import TrackWorld
    from f1rl.sandbox.track.state import TrackState
    from f1rl.sandbox.vehicle import VehicleControl, VehicleParameters
    from f1rl.sandbox.world import WorldState, step_world

    patch,q=surface(pi/6.)
    initial=car(patch,q)
    world=WorldState(initial,TrackState(frozenset()),
                     InteractionState(frozenset(),frozenset()),ScoreState())
    # Zero max speed avoids adding uncontrolled downhill translation in this
    # isolated contact-certification test; this is not a training setup.
    params=VehicleParameters(2.5,.3,.6,0.,20.,0.,0.,0.,9.81)
    args=(TrackWorld(surfaces=(patch,)),world,
          VehicleControl(0.,0.,0.),params,.02,.5)
    with pytest.raises(RuntimeError,match="road validity unsupported"):
        step_world(*args,body_geometry=BODY)
    result=step_world(*args,body_geometry=BODY,
        ramp_support_policy=PlanarRampPolicy(WHEELS,TOLERANCE))
    assert result.state.vehicle.grounded
    assert result.state.vehicle.position==initial.position
    assert result.events==()


def test_world_rejects_multisurface_seam_without_support_rule():
    from f1rl.sandbox.interactions.state import InteractionState
    from f1rl.sandbox.rules.scoring import ScoreState
    from f1rl.sandbox.ramp_support import PlanarRampPolicy
    from f1rl.sandbox.track.model import TrackWorld
    from f1rl.sandbox.track.state import TrackState
    from f1rl.sandbox.vehicle import VehicleControl, VehicleParameters
    from f1rl.sandbox.world import WorldState, step_world

    patch,q=surface(pi/6.)
    initial=car(patch,q)
    world=WorldState(initial,TrackState(frozenset()),
                     InteractionState(frozenset(),frozenset()),ScoreState())
    params=VehicleParameters(2.5,.3,.6,0.,20.,0.,0.,0.,9.81)
    extra=PlanarSurfacePatch("second",patch.origin,patch.forward,
                             patch.normal,patch.width,patch.length)
    with pytest.raises(RuntimeError,match="requires exactly one planar patch"):
        step_world(TrackWorld(surfaces=(patch,extra)),world,
            VehicleControl(0.,0.,0.),params,.02,.5,
            body_geometry=BODY,ramp_support_policy=PlanarRampPolicy(WHEELS,TOLERANCE))


def test_world_rejects_ramp_policy_without_body():
    from f1rl.sandbox.interactions.state import InteractionState
    from f1rl.sandbox.rules.scoring import ScoreState
    from f1rl.sandbox.ramp_support import PlanarRampPolicy
    from f1rl.sandbox.track.model import TrackWorld
    from f1rl.sandbox.track.state import TrackState
    from f1rl.sandbox.vehicle import VehicleControl, VehicleParameters
    from f1rl.sandbox.world import WorldState, step_world
    patch,q=surface(pi/6.)
    initial=car(patch,q)
    world=WorldState(initial,TrackState(frozenset()),
                     InteractionState(frozenset(),frozenset()),ScoreState())
    params=VehicleParameters(2.5,.3,.6,0.,20.,0.,0.,0.,9.81)
    with pytest.raises(ValueError,match="ramp_support_policy requires"):
        step_world(TrackWorld(surfaces=(patch,)),world,
            VehicleControl(0.,0.,0.),params,.02,.5,
            ramp_support_policy=PlanarRampPolicy(WHEELS,TOLERANCE))


@pytest.mark.parametrize("value",[0.,-1.,float("nan"),True])
def test_policy_rejects_unreviewable_contact_tolerance(value):
    from f1rl.sandbox.ramp_support import PlanarRampPolicy
    with pytest.raises(ValueError,match="tolerance"):
        PlanarRampPolicy(WHEELS,value)
