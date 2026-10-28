"""Microstep swept quaternion paths must preserve source endpoints exactly."""
from math import pi

from src.simulation.body_collision import _slerp
from src.simulation.geometry.rotation import Quaternion
from src.simulation.geometry.vector import Vec3


def test_slerp_microstep_has_correct_source_endpoints_and_interior():
    first=Quaternion.identity()
    second=Quaternion.from_axis_angle(Vec3(1.,0.,0.),2.1e-6)
    dot=first.w*second.w+first.x*second.x+first.y*second.y+first.z*second.z
    assert dot>1.-1e-12
    assert _slerp(first,second,0.)==first
    assert _slerp(first,second,1.)==second
    middle=_slerp(first,second,.5)
    assert middle!=first
    assert middle!=second
    wheel=Vec3(0.,0.,1.)
    y0=first.rotate(wheel).y
    ym=middle.rotate(wheel).y
    y1=second.rotate(wheel).y
    assert y0==0.
    assert y0>ym>y1
    assert abs(ym-y1/2.)<1e-9


def test_slerp_ordinary_quarter_turn_unchanged():
    first=Quaternion.identity()
    second=Quaternion.from_axis_angle(Vec3(0.,1.,0.),pi/2.)
    mid=_slerp(first,second,.5)
    assert abs(mid.rotate(Vec3(0.,0.,1.)).x-2**-.5)<1e-10
    assert _slerp(first,second,0.)==first
    assert _slerp(first,second,1.)==second
