"""Fail-closed ballistic CCD geometry tests, not scored simulator evaluation."""
from dataclasses import replace
import pytest
from f1rl.sandbox.ballistic_body_impact import first_ballistic_body_impact
from f1rl.sandbox.body_impact import first_body_impact_bracket
from f1rl.sandbox.body_collision import VehicleBodyGeometry
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.track.volume import OrientedBoxVolume
from f1rl.sandbox.vehicle import VehicleState

ZERO=Vec3(0.,0.,0.)
BODY=VehicleBodyGeometry(.2,.2,.2,ZERO)
OBSTACLE=OrientedBoxVolume(Vec3(0.,1.,0.),Vec3(.05,.1,.5),Quaternion.identity())


def pose(x,y,z=0.,vx=0.,vy=0.,q=None):
    return VehicleState(Vec3(x,y,z),Vec3(vx,vy,0.),
                        q or Quaternion.identity(),ZERO,False)


def parabola_trajectory():
    # Specified test fixture: x=-2+4t, y=4t-4t^2, t in [0,1].
    # The straight chord has y=0 and misses a body obstacle at y=1.
    previous=pose(-2.,0.,vx=4.,vy=4.)
    terminal=pose(2.,0.,vx=4.,vy=-4.)
    return previous,terminal,Vec3(0.,-8.,0.)


def test_curved_ballistic_path_catches_contact_linear_chord_misses():
    first,last,a=parabola_trajectory()
    assert first_body_impact_bracket(first,last,BODY,OBSTACLE,duration_s=1.) is None
    hit=first_ballistic_body_impact(first,last,BODY,OBSTACLE,
                                    duration_s=1.,acceleration_world=a,
                                    time_tolerance_s=2e-5)
    assert hit is not None
    assert hit.version=="ballistic_body_first_witness_v1"
    assert .45<hit.last_proven_clear_s<hit.first_witnessed_contact_by_s<.5
    assert hit.uncertainty_s<=2e-5
    assert hit.gap_checks>0


def test_far_obstacle_yields_verified_clear_not_false_positive():
    first,last,a=parabola_trajectory()
    far=OrientedBoxVolume(Vec3(0.,10.,0.),Vec3(.05,.1,.5),Quaternion.identity())
    assert first_ballistic_body_impact(first,last,BODY,far,
                                      duration_s=1.,acceleration_world=a) is None


def test_high_speed_thin_obstacle_sweep_witnessed():
    first=pose(-10.,0.,vx=20.)
    last=pose(10.,0.,vx=20.)
    thin=OrientedBoxVolume(Vec3(0.,0.,0.),Vec3(.0005,.25,.25),Quaternion.identity())
    hit=first_ballistic_body_impact(first,last,BODY,thin,duration_s=1.,
                                    acceleration_world=ZERO,time_tolerance_s=2e-6,
                                    max_refinements=20)
    assert hit and .49<hit.last_proven_clear_s<hit.first_witnessed_contact_by_s<.5


def test_registered_ballistic_endpoint_mismatch_rejected():
    first,last,a=parabola_trajectory()
    with pytest.raises(ValueError,match="endpoint"):
        first_ballistic_body_impact(first,replace(last,position=last.position+Vec3(0.,.1,0.)),
                                    BODY,OBSTACLE,duration_s=1.,acceleration_world=a)


def test_initial_overlap_rejected():
    first=pose(0.,1.,vx=0.)
    with pytest.raises(ValueError,match="already overlapping"):
        first_ballistic_body_impact(first,first,BODY,OBSTACLE,
                                    duration_s=1.,acceleration_world=ZERO)


@pytest.mark.parametrize("value",[0.,-1.,True,float("nan"),float("inf")])
def test_bad_time_fails_closed(value):
    first,last,a=parabola_trajectory()
    with pytest.raises(ValueError):
        first_ballistic_body_impact(first,last,BODY,OBSTACLE,
                                    duration_s=value,acceleration_world=a)


def test_too_coarse_refinement_rejected_not_falsely_cleared():
    first,last,a=parabola_trajectory()
    with pytest.raises(ValueError,match="unresolvable"):
        first_ballistic_body_impact(first,last,BODY,OBSTACLE,duration_s=1.,
                                    acceleration_world=a,max_refinements=1,
                                    time_tolerance_s=1e-8)
