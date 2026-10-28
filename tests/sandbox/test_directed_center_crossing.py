"""Synthetic directed centre crossing tests, no race-state integration yet."""
from math import pi

import pytest

from src.simulation.directed_center_crossing import directed_center_gate_crossing
from src.simulation.geometry.rotation import Quaternion
from src.simulation.geometry.vector import Vec3
from src.simulation.track.volume import OrientedBoxVolume


def gate(*, orientation=Quaternion.identity(),x=1.,y=1.,depth=.01):
    return OrientedBoxVolume(Vec3(0.,0.,0.),Vec3(x,y,depth),orientation)


def test_forward_sweep_hits_arbitrarily_thin_gate():
    hit=directed_center_gate_crossing(Vec3(0.,0.,-10.),Vec3(0.,0.,10.),gate(depth=.00001))
    assert hit is not None
    assert hit.fraction == pytest.approx(.5)
    assert hit.center==Vec3(0.,0.,0.)


def test_reverse_direction_never_counts():
    assert directed_center_gate_crossing(Vec3(0.,0.,5.),Vec3(0.,0.,-5.),gate()) is None


def test_nonintersecting_line_outside_gate_xy_is_rejected():
    assert directed_center_gate_crossing(Vec3(2.,0.,-5.),Vec3(2.,0.,5.),gate()) is None
    assert directed_center_gate_crossing(Vec3(0.,2.,-5.),Vec3(0.,2.,5.),gate()) is None


def test_already_inside_or_on_gate_plane_does_not_retrigger():
    assert directed_center_gate_crossing(Vec3(0.,0.,0.),Vec3(0.,0.,5.),gate()) is None
    assert directed_center_gate_crossing(Vec3(0.,0.,.1),Vec3(0.,0.,5.),gate()) is None


def test_rotated_gate_respects_its_authored_forward_direction():
    rotation=Quaternion.from_axis_angle(Vec3(0.,1.,0.),pi/2.)
    g=gate(orientation=rotation)
    assert directed_center_gate_crossing(Vec3(-5.,0.,0.),Vec3(5.,0.,0.),g) is not None
    assert directed_center_gate_crossing(Vec3(5.,0.,0.),Vec3(-5.,0.,0.),g) is None


def test_oblique_segment_intersects_gate():
    hit=directed_center_gate_crossing(Vec3(-2.,0.,-4.),Vec3(2.,0.,4.),gate(x=.5))
    assert hit is not None
    assert hit.center.x == pytest.approx(0.)


def test_parallel_motion_not_a_crossing():
    assert directed_center_gate_crossing(Vec3(0,0,-2),Vec3(.1,0,-2),gate()) is None


@pytest.mark.parametrize("bad",[-1.,float("nan"),1.,True])
def test_tolerance_contract_is_strict(bad):
    with pytest.raises(ValueError,match="tolerance"):
        directed_center_gate_crossing(Vec3(0,0,-1),Vec3(0,0,1),gate(),tolerance_m=bad)
