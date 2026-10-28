"""Provably bounded first linear+SLERP body contact, not scored Eval."""
from dataclasses import replace

import pytest
from src.sandbox.body_collision import VehicleBodyGeometry
from src.sandbox.geometry.vector import Vec3
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.linear_first_body_contact import certify_first_linear_body_contact
from src.sandbox.track.volume import OrientedBoxVolume
from src.sandbox.vehicle import VehicleState

ZERO=Vec3(0.,0.,0.)
BODY=VehicleBodyGeometry(.4,.4,.4,ZERO)
OBSTACLE=OrientedBoxVolume(Vec3(0.,0.,0.),Vec3(.01,.5,.5),Quaternion.identity())


def car(x:float,v=0.):
    return VehicleState(Vec3(x,0.,0.),Vec3(v,0.,0.),
        Quaternion.identity(),ZERO,True)


def test_witness_brackets_first_thin_obstacle_contact():
    hit=certify_first_linear_body_contact(
        car(-1.),car(1.),BODY,OBSTACLE,
        duration_s=.2,max_refinements=20,resolution_s=1e-6)
    assert hit is not None
    assert .078<hit.last_certified_clear_s<hit.first_witnessed_overlap_s<.080
    assert hit.uncertainty_s<=1e-6
    assert hit.gap_checks>=1


def test_direction_reversal_tracks_earliest_contact_from_other_side():
    hit=certify_first_linear_body_contact(car(1.),car(-1.),BODY,OBSTACLE,
                                          duration_s=.2,max_refinements=20)
    assert hit is not None
    assert .078<hit.last_certified_clear_s<hit.first_witnessed_overlap_s<.080


def test_stationary_far_clear_no_fictitious_contact():
    assert certify_first_linear_body_contact(car(-10.),car(-9.),BODY,OBSTACLE,
        duration_s=.2) is None


def test_initial_overlap_fails_closed():
    with pytest.raises(ValueError,match="overlapping"):
        certify_first_linear_body_contact(car(0.),car(1.),BODY,OBSTACLE,duration_s=.2)


@pytest.mark.parametrize("duration",[0.,-1.,float("nan"),True])
def test_invalid_time_rejected(duration):
    with pytest.raises(ValueError):
        certify_first_linear_body_contact(car(-1.),car(1.),BODY,OBSTACLE,duration_s=duration)


def test_ambiguous_grazing_refuses_to_claim_witness():
    # Degenerate almost-touch of the body is uncertain at coarse resolution.
    far=OrientedBoxVolume(Vec3(0.,.700000001,0.),Vec3(.01,.5,.5),Quaternion.identity())
    try:
        witness=certify_first_linear_body_contact(car(-1.),car(1.),BODY,far,
            duration_s=.2,resolution_s=1e-5,max_refinements=20)
    except ValueError:
        return
    assert witness is None
