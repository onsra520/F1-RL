"""Exact finite OBB closest feature witnesses for source-bounded road edges."""
from math import pi
import pytest
from src.simulation.geometry.vector import Vec3
from src.simulation.geometry.rotation import Quaternion
from src.simulation.track.volume import OrientedBoxVolume
from src.simulation.obb_feature_contact import closest_obb_features,closest_point
from src.simulation.body_collision import obb_separation_lower_bound
Q=Quaternion.identity()

def box(x,y,z,hx=.5,hy=.5,hz=.5,q=Q):
    return OrientedBoxVolume(Vec3(x,y,z),Vec3(hx,hy,hz),q)

@pytest.mark.parametrize("axis",[(1.1,0.,0.),(0.,1.1,0.),(0.,0.,1.1)])
def test_separated_axis_aligned_boxes_achieve_exact_geometric_gap(axis):
    body,road=box(*axis),box(0.,0.,0.)
    witness=closest_obb_features(body,road)
    assert witness.separation_m==pytest.approx(.1,abs=1e-9)
    assert (witness.body_point-witness.obstacle_point).magnitude==pytest.approx(.1)
    assert obb_separation_lower_bound(body,road)==pytest.approx(.1)

def test_angled_skew_boxes_witness_actual_feature_points():
    q=Quaternion.from_axis_angle(Vec3(0.,1.,0.),pi/4.)
    car=box(1.65,0.,0.,q=q)
    road=box(0.,0.,0.)
    witness=closest_obb_features(car,road)
    assert witness.separation_m>0.
    assert (witness.body_point-closest_point(car,witness.body_point)).magnitude<1e-8
    assert (witness.obstacle_point-closest_point(road,witness.obstacle_point)).magnitude<1e-8
    assert witness.separation_m >= obb_separation_lower_bound(car,road)-1e-8

def test_overlap_is_not_misrepresented_as_clear():
    with pytest.raises(ValueError,match="intersect"):
        closest_obb_features(box(0.,0.,0.),box(.1,0.,0.))
