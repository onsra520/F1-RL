import pytest

from src.simulation.center_trajectory import CenterSegment
from src.simulation.geometry.vector import Vec3
from src.simulation.geometry.rotation import Quaternion
from src.simulation.track.volume import OrientedBoxVolume


def test_ballistic_gate_uses_exact_height_instead_of_endpoint_chord():
    segment = CenterSegment(0.,1.,Vec3(0.,1.,-1.),Vec3(0.,2.,1.),
                            Vec3(0.,2.,2.),2.)
    gate = OrientedBoxVolume(Vec3(0.,1.75,0.),Vec3(1.,.01,.001),Quaternion.identity())
    assert segment.directed_crossings(gate) == pytest.approx((.5,))
    assert segment.position_at(.5) == Vec3(0.,1.75,0.)


def test_nonphysical_ballistic_endpoint_is_rejected():
    with pytest.raises(ValueError,match="endpoint"):
        CenterSegment(0.,1.,Vec3(0.,1.,-1.),Vec3(0.,99.,1.),Vec3(0.,2.,2.),2.)


def test_rotated_gate_preserves_direction_and_tick_local_time():
    segment = CenterSegment(2.,3.,Vec3(-1.,0.,0.),Vec3(1.,0.,0.))
    gate = OrientedBoxVolume(Vec3(0.,0.,0.),Vec3(1.,1.,.001),
                             Quaternion.from_axis_angle(Vec3(0.,1.,0.),1.5707963267948966))
    assert segment.directed_crossings(gate) == pytest.approx((2.5,))
    assert CenterSegment(2.,3.,segment.end,segment.start).directed_crossings(gate) == ()


def test_two_ballistic_plane_roots_only_credit_forward_crossing():
    segment = CenterSegment(0.,3.,Vec3(0.,-1.,0.),Vec3(0.,-1.,0.),Vec3(0.,3.,0.),2.)
    gate = OrientedBoxVolume(Vec3(0.,0.,0.),Vec3(10.,10.,.001),
        Quaternion.from_axis_angle(Vec3(1.,0.,0.),-1.5707963267948966))
    assert segment.directed_crossings(gate) == pytest.approx((.3819660112501051,))


def test_tangent_contact_and_starting_on_plane_do_not_earn_crossing():
    gate = OrientedBoxVolume(Vec3(0.,0.,0.),Vec3(10.,10.,.001),
        Quaternion.from_axis_angle(Vec3(1.,0.,0.),-1.5707963267948966))
    assert CenterSegment(0.,2.,Vec3(0.,-1.,0.),Vec3(0.,-1.,0.),
                         Vec3(0.,2.,0.),2.).directed_crossings(gate) == ()
    gate = OrientedBoxVolume(Vec3(0.,0.,0.),Vec3(1.,1.,.001),Quaternion.identity())
    assert CenterSegment(0.,1.,Vec3(0.,0.,0.),Vec3(0.,0.,1.)).directed_crossings(gate) == ()


def test_rounded_endpoint_root_is_credited_once_across_segments():
    gate = OrientedBoxVolume(Vec3(0.,0.,0.),Vec3(1.,1.,.001),Quaternion.identity())
    incoming = CenterSegment(0.,.03,Vec3(0.,0.,-2.3),Vec3(0.,0.,0.))
    outgoing = CenterSegment(.03,.06,Vec3(0.,0.,0.),Vec3(0.,0.,2.3))
    assert incoming.directed_crossings(gate) == pytest.approx((.03,))
    assert outgoing.directed_crossings(gate) == ()
