"""Independent 3D exact segment / conservative swept-body geometry tests."""
from math import pi, sqrt

import pytest

from src.rl.swept_volume_audit import (
    BodyProxy, segment_to_oriented_box_distance_3d,
    swept_body_proxy_clearance, PROXY_PROFILES,
)
from src.simulation.geometry.vector import Vec3
from src.simulation.geometry.rotation import Quaternion
from src.simulation.track.volume import OrientedBoxVolume


def box(orientation=None):
    return OrientedBoxVolume(Vec3(0.,0.,0.), Vec3(1.,1.,1.),orientation or Quaternion.identity())


@pytest.mark.parametrize("a,b,expected",[
    ((-3,0,0),(3,0,0),0.),    # continuous collision; both discrete endpoints clear
    ((0,2,0),(0,2,0),1.),    # stationary
    ((-3,2,0),(3,2,0),1.),    # face parallel
    ((2,2,2),(3,3,3),sqrt(3)), # corner
    ((3,0,0),(2,0,0),1.),    # face
    ((0,0,0),(0,0,0),0.),    # inside
    ((-3,3,-3),(-1,1,-1),0.), # touches corner
])
def test_exact_segment_box_3d(a,b,expected):
    assert segment_to_oriented_box_distance_3d(Vec3(*a),Vec3(*b),box())==pytest.approx(expected)


def test_3d_overflight_not_false_positive_from_xz_projection():
    assert segment_to_oriented_box_distance_3d(
        Vec3(-3.,3.,0.),Vec3(3.,3.,0.),box()
    )==pytest.approx(2.)


def test_rotated_obstacle_respects_local_frame():
    rotation=Quaternion.from_axis_angle(Vec3(0.,1.,0.),pi/2)
    volume=OrientedBoxVolume(
        Vec3(4.,1.,-2.),Vec3(.5,1.,2.),rotation
    )
    # Local (3,0,0) rotates to world (0,0,-3): nearest x-face distance 2.5
    first=volume.center + rotation.rotate(Vec3(3.,0.,0.))
    assert segment_to_oriented_box_distance_3d(first,first,volume)==pytest.approx(2.5)
    assert segment_to_oriented_box_distance_3d(
        volume.center + rotation.rotate(Vec3(-3.,0.,0.)),
        volume.center + rotation.rotate(Vec3(3.,0.,0.)),
        volume,
    )==pytest.approx(0.)


def test_geom_translation_and_endpoint_reversal():
    original=segment_to_oriented_box_distance_3d(
        Vec3(-2.,2.2,0.),Vec3(2.,2.2,0.),box()
    )
    shift=Vec3(5.,-8.,2.)
    moved=OrientedBoxVolume(shift,Vec3(1.,1.,1.),Quaternion.identity())
    shifted=segment_to_oriented_box_distance_3d(
        Vec3(-2.,2.2,0.)+shift,Vec3(2.,2.2,0.)+shift,moved
    )
    reversed_=segment_to_oriented_box_distance_3d(
        Vec3(2.,2.2,0.),Vec3(-2.,2.2,0.),box()
    )
    assert original==pytest.approx(1.2)
    assert shifted==pytest.approx(original)
    assert reversed_==pytest.approx(original)


def test_assumed_envelopes_have_positive_dimensions_and_are_distinct():
    assert len(set(x.name for x in PROXY_PROFILES))==3
    assert PROXY_PROFILES[0].length_m>=3.0
    with pytest.raises(ValueError,match="positive"):
        BodyProxy("bad",-1.,1.,1.,0.)
    with pytest.raises(ValueError,match="stations"):
        BodyProxy("bad",3.,1.,1.,0.,longitudinal_stations=1)


def test_swept_proxy_far_clear_and_crossing_inconclusive():
    proxy=BodyProxy("test",2.,.8,.8,0.,longitudinal_stations=5)
    q=Quaternion.identity()
    clear=swept_body_proxy_clearance(
        Vec3(-3.,4.,0.),q,Vec3(3.,4.,0.),q,box(),proxy
    )
    assert clear>2.0
    inconclusive=swept_body_proxy_clearance(
        Vec3(-3.,0.,0.),q,Vec3(3.,0.,0.),q,box(),proxy
    )
    assert inconclusive<0.0


def test_station_rotation_sagitta_conservative_for_half_turn():
    q0=Quaternion.identity()
    q1=Quaternion.from_axis_angle(Vec3(0,1,0),pi)
    proxy=BodyProxy("rotation",4.,.5,.5,0.,longitudinal_stations=3)
    # The middle of a half-turn swings the front station toward +/-X,
    # although its endpoints are separated on Z. This should not be cleared
    # simply by checking the chord without an angular rotation margin.
    obstacle=OrientedBoxVolume(
        Vec3(2.,0.,0.),Vec3(.2,.2,.2),Quaternion.identity()
    )
    conservative_gap=swept_body_proxy_clearance(
        Vec3(0.,0.,0.),q0,Vec3(0.,0.,0.),q1,obstacle,proxy
    )
    assert conservative_gap<0.0


def test_body_proxy_requires_real_footprint_review_even_when_pass():
    proxy=PROXY_PROFILES[0]
    assert proxy.cross_section_radius_m>proxy.width_m/2.



def test_swept_envelope_never_exceeds_sampled_body_corner_clearance():
    """Numerically cross-check conservative lower bound over rotation interpolation.

    The interpolated sample checks are only validation; the audit uses its
    analytical station + sagitta envelope, not discrete collision sampling.
    """
    from math import cos, sin
    q0=Quaternion.identity()
    p0=Vec3(-.5,0.,-.2)
    p1=Vec3(.4,.15,.7)
    proxy=BodyProxy("sensitivity",3.8,1.6,1.2,.45,longitudinal_stations=9)
    box_volume=OrientedBoxVolume(
        Vec3(2.0,.8,.8),Vec3(.5,.5,.6),
        Quaternion.from_axis_angle(Vec3(0.,1.,0.),.3),
    )
    for angle in (.0,.15,.8,pi/2,pi):
        q1=Quaternion.from_axis_angle(Vec3(0.,1.,0.),angle) if angle else q0
        lower=swept_body_proxy_clearance(p0,q0,p1,q1,box_volume,proxy)
        observed=float("inf")
        for tick in range(41):
            t=tick/40.
            pos=p0*(1.-t)+p1*t
            turn=Quaternion.from_axis_angle(Vec3(0.,1.,0.),angle*t) if angle*t else q0
            for length in (-proxy.length_m/2,0.,proxy.length_m/2):
                for x in (-proxy.width_m/2,proxy.width_m/2):
                    for y in (
                        proxy.center_y_offset_m-proxy.height_m/2,
                        proxy.center_y_offset_m+proxy.height_m/2,
                    ):
                        local=Vec3(x,y,length)
                        world=pos+turn.rotate(local)
                        actual=segment_to_oriented_box_distance_3d(world,world,box_volume)
                        observed=min(observed,actual)
        assert lower<=observed+1e-7, (angle,lower,observed)
