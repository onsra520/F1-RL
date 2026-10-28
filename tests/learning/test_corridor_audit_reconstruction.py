"""Geometric regression for continuous union-of-road-patches coverage."""
from math import sqrt

import pytest

from src.learning.corridor_audit import (
    clip_xz_segment_to_patch,continuous_union_road_coverage,
)
from src.sandbox.geometry.vector import Vec3
from src.sandbox.track.patch import PlanarSurfacePatch


def road(name,origin,forward,width,length):
    return PlanarSurfacePatch(
        name,Vec3(*origin),Vec3(*forward),Vec3(0.,1.,0.),
        width,length,
    )


ENTRY=road("entry",(0,0,35),(0,0,1),12,70)
EXIT=road("exit",(35,0,70),(1,0,0),12,70)


@pytest.mark.parametrize("first,second,covered,outside",[
    ((0,0,4),(0,0,20),16,0),
    ((0,0,69),(20,0,69),20,0), # crosses shared intersection without gap
    ((-10,0,50),(10,0,50),12,8),
    ((-10,0,80),(10,0,80),0,20),
    ((-8,0,70),(72,0,70),76,4), # entry [-6,6] union exit [0,70]
    ((-20,0,35),(20,0,35),12,28),
    ((0,0,60),(10,0,60),6,4),
])
def test_exact_continuous_centerline_road_union(first,second,covered,outside):
    c,u=continuous_union_road_coverage(Vec3(*first),Vec3(*second),(ENTRY,EXIT))
    assert c==pytest.approx(covered)
    assert u==pytest.approx(outside)


def test_diagonal_segment_between_discrete_inside_endpoints_has_gap():
    # From entry x=0 z=55 to exit x=18 z=70; near the corner the union
    # permits some parts of the diagonal but not a shortcut across outer ground.
    a=Vec3(0,0,55)
    b=Vec3(18,0,70)
    length=sqrt(18**2+15**2)
    covered,uncovered=continuous_union_road_coverage(a,b,(ENTRY,EXIT))
    assert covered+uncovered==pytest.approx(length)
    # This specific path stays on entry until z>=64 when x=10.8,
    # hence cuts through a non-road wedge near the junction.
    assert uncovered>0


def test_two_disjoint_patches_do_not_fill_outside_middle():
    left=road("left",(-10,0,0),(0,0,1),2,10)
    right=road("right",(10,0,0),(0,0,1),2,10)
    covered,uncovered=continuous_union_road_coverage(
        Vec3(-15,0,0),Vec3(15,0,0),(left,right))
    assert covered==pytest.approx(4.)
    assert uncovered==pytest.approx(26.)


def test_patch_orientation_different_local_forward_supported():
    north=road("north",(0,0,0),(0,0,-1),6,20)
    c,u=continuous_union_road_coverage(Vec3(-4,0,0),Vec3(4,0,0),(north,))
    assert c==pytest.approx(6.)
    assert u==pytest.approx(2.)


def test_ramp_patch_is_rejected_not_silently_flattened():
    patch=PlanarSurfacePatch(
        "ramp",Vec3(0,0,0),
        Vec3(0,1/sqrt(2),1/sqrt(2)),
        Vec3(0,1/sqrt(2),-1/sqrt(2)),8.,20.,
    )
    with pytest.raises(ValueError,match="horizontal"):
        clip_xz_segment_to_patch(Vec3(0,0,0),Vec3(0,0,4),patch)


def test_patch_boundaries_are_closed():
    assert clip_xz_segment_to_patch(Vec3(-6,0,35),Vec3(-6,0,35),ENTRY)==pytest.approx((0.,1.))
    assert clip_xz_segment_to_patch(Vec3(-7,0,35),Vec3(-7,0,35),ENTRY) is None


def test_empty_patches_and_invalid_endpoints_rejected():
    with pytest.raises(ValueError,match="nonempty"):
        continuous_union_road_coverage(Vec3(0,0,0),Vec3(1,0,0),())
    with pytest.raises(ValueError,match="Vec3"):
        continuous_union_road_coverage((0,0,0),Vec3(1,0,0),(ENTRY,))
