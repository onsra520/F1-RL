"""WP7 TRAIN-only swept-point-to-obstacle geometry; no HELD-OUT evaluation."""
from math import sqrt

import pytest

from src.rl.trajectory_audit import segment_rect_clearance_xz


@pytest.mark.parametrize("a,b,expected", [
    ((-2.0,0.0),(2.0,0.0),0.0),     # tunnelling: both discrete endpoints outside
    ((-2.0,2.0),(2.0,2.0),1.0),     # passes above rectangular face in XZ
    ((0.0,2.0),(0.0,2.0),1.0),      # stationary vehicle
    ((2.0,2.0),(3.0,3.0),sqrt(2.0)),  # corner outside
    ((-3.0,-3.0),(-1.0,-1.0),0.0), # touches corner
    ((1.0,0.0),(1.0,0.0),0.0),     # on edge
    ((3.0,0.0),(2.0,0.0),1.0),     # near face
])
def test_continuous_segment_rectangle_distance(a,b,expected):
    assert segment_rect_clearance_xz(*a,*b,0.0,0.0,1.0,1.0)==pytest.approx(expected)


def test_swept_geometry_reversal_and_translation():
    a=(-3.,2.3)
    b=(3.,2.3)
    original=segment_rect_clearance_xz(*a,*b,0.,0.,1.,1.)
    reverse=segment_rect_clearance_xz(*b,*a,0.,0.,1.,1.)
    translated=segment_rect_clearance_xz(a[0]+10,a[1]-5,b[0]+10,b[1]-5,10.,-5.,1.,1.)
    assert original==pytest.approx(1.3)
    assert reverse==pytest.approx(original)
    assert translated==pytest.approx(original)


@pytest.mark.parametrize("data", [
    (-2.,0.,2.,0.,0.,0.,0.,1.),
    (-2.,0.,2.,0.,0.,0.,1.,float("nan")),
    (-2.,0.,2.,0.,0.,0.,1.,float("inf")),
])
def test_bad_or_nonfinite_geometry_fails_closed(data):
    with pytest.raises(ValueError,match="geometry"):
        segment_rect_clearance_xz(*data)
