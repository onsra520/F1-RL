"""Prove bounded finite road-surface intersection vs infinite plane false positive."""
from dataclasses import replace
import pytest

from f1rl.sandbox.body_collision import VehicleBodyGeometry
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.independent_suspension_dynamics import SuspensionResolutionError
from f1rl.sandbox.multi_surface_support import _atlas
from f1rl.sandbox.swept_surface_clearance import (
    certify_swept_chassis_clearance,certify_swept_chassis_patch_intersection_free,
)
from f1rl.sandbox.track.patch import PlanarSurfacePatch
from f1rl.sandbox.vehicle import VehicleState

BODY=VehicleBodyGeometry(2.,.2,2.,Vec3(0.,0.,0.))
ROAD=PlanarSurfacePatch("finite",Vec3(0.,0.,0.),Vec3(0.,0.,1.),Vec3(0.,1.,0.),1.,20.)
ATLAS=_atlas((ROAD,),1e-6)


def state(x,y,tilt):
    return VehicleState(Vec3(x,y,0.),Vec3(0.,0.,0.),
        Quaternion.from_axis_angle(Vec3(0.,0.,1.),tilt),Vec3(0.,0.,0.),True)


def test_off_road_lowest_body_corner_is_not_witnessed_finite_patch_collision():
    # Body left tip extends over x ~ .4 of road, lifted well above y=0;
    # lowest rotated right corner hangs below road plane OUTSIDE road width.
    p=state(1.2,.35,-.7)
    with pytest.raises(SuspensionResolutionError,match="penetration"):
        certify_swept_chassis_clearance(lambda t:p,0.,0.,BODY,ATLAS)
    assert certify_swept_chassis_patch_intersection_free(
        lambda t:p,0.,0.,BODY,ATLAS)>=1
    assert certify_swept_chassis_patch_intersection_free(
        lambda t:p,0.,1.,BODY,ATLAS)>=1


def test_real_box_patch_intersection_is_never_falsely_cleared():
    p=state(0.,.05,0.)
    with pytest.raises(SuspensionResolutionError,match="physical road contact"):
        certify_swept_chassis_patch_intersection_free(lambda t:p,0.,1.,BODY,ATLAS)


def test_swept_contact_between_two_clear_endpoints_remains_blocked():
    first=state(-3.,.02,0.)
    def path(t):
        return replace(first,position=Vec3(-3.+6*t,.02,0.))
    with pytest.raises(SuspensionResolutionError):
        certify_swept_chassis_patch_intersection_free(path,0.,1.,BODY,ATLAS)


def test_certification_rejects_ambiguous_refinement_budget():
    p=state(-1.51,.201,0.)
    def rapid_clear_sweep(t):
        return replace(p,position=Vec3(-1.51+3.02*t,.201,0.))
    with pytest.raises(SuspensionResolutionError,match="unresolved"):
        certify_swept_chassis_patch_intersection_free(rapid_clear_sweep,0.,1.,BODY,ATLAS,max_subdivisions=0)


def test_bounded_road_first_contact_time_is_witnessed_not_inferred_from_corner():
    from f1rl.sandbox.swept_surface_clearance import certify_first_bounded_road_contact
    first=state(0.,.30,0.)
    after=state(0.,.05,0.)
    event=certify_first_bounded_road_contact(first,after,BODY,ATLAS,
                                             duration_s=.1,resolution_s=1e-6)
    assert event is not None
    patch,lower,upper,checks=event
    assert patch=="finite"
    assert .078<lower<upper<.082
    assert upper-lower<=1e-6
    assert checks>0


def test_bounded_road_first_contact_ignores_low_overhang_off_patch():
    from f1rl.sandbox.swept_surface_clearance import certify_first_bounded_road_contact
    p=state(1.2,.35,-.7)
    assert certify_first_bounded_road_contact(p,p,BODY,ATLAS,duration_s=.1) is None
