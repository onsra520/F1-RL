"""Source-bound micro-roundoff road-crash prefix regression."""
from dataclasses import replace

from src.core.road_crash_validation import same_reconstructed_road_pose
from src.simulation.vehicle import VehicleState
from src.simulation.geometry.vector import Vec3
from src.simulation.geometry.rotation import Quaternion

def sample():
    return VehicleState(Vec3(.2,.4,2.),Vec3(2.,-.1,8.),
                        Quaternion.identity(),Vec3(.2,.3,.4),True)

def test_real_inertia_roundoff_keeps_same_accepted_state():
    a=sample()
    b=replace(a,angular_velocity=Vec3(.2+1e-16,.3-1e-16,.4+1e-16))
    assert a!=b
    assert same_reconstructed_road_pose(a,b)

def test_reconstructed_pose_never_allows_momentum_physics_drift():
    a=sample()
    assert not same_reconstructed_road_pose(a,replace(
        a,angular_velocity=Vec3(.2+1e-8,.3,.4)))
    assert not same_reconstructed_road_pose(a,replace(
        a,velocity=Vec3(2.+1e-12,-.1,8.)))
    assert not same_reconstructed_road_pose(a,replace(
        a,position=Vec3(.2+1e-8,.4,2.)))
    assert not same_reconstructed_road_pose(a,replace(a,grounded=False))


def test_exact_frozen_b2_slalom_road_crash_source_quaternion_one_ulp():
    # B-seed2 / slalom nominal / e4 contains a one-ULP normalization difference.
    from dataclasses import replace
    from math import nextafter, inf
    a=sample()
    b=replace(a)
    source_q=object.__new__(Quaternion)
    for key in ("w","x","y","z"):
        object.__setattr__(source_q,key,nextafter(getattr(a.orientation,key),inf))
    # Avoid extra Quaternion constructor normalization *only in this test*.
    object.__setattr__(b,"orientation",source_q)
    assert a.orientation!=b.orientation
    assert same_reconstructed_road_pose(a,b)
    forged_q=object.__new__(Quaternion)
    for key in ("w","x","y","z"):
        object.__setattr__(forged_q,key,
            getattr(a.orientation,key)+(1e-8 if key=="w" else 0.))
    object.__setattr__(b,"orientation",forged_q)
    assert not same_reconstructed_road_pose(a,b)
