"""Measured wheel-hub contact geometry and calibration provenance contracts."""
from dataclasses import replace
import pytest

from src.simulation.tire_calibration_contract import (
    TireHubGeometry,sample_spherical_tire_contact,
    MeasuredPhysicsParameter,PhysicsCalibrationRegister,
    REQUIRED_WP7_MEASUREMENTS,
)
from src.simulation.geometry.vector import Vec3
from src.simulation.geometry.rotation import Quaternion
from src.simulation.track.patch import PlanarSurfacePatch

HUBS=(
    Vec3(-.4,0.,.8),Vec3(.4,0.,.8),
    Vec3(-.4,0.,-.8),Vec3(.4,0.,-.8),
)
GEO=TireHubGeometry(HUBS,.3,.2)
FLAT=PlanarSurfacePatch(
    "flat",Vec3(0.,0.,0.),Vec3(0.,0.,1.),Vec3(0.,1.,0.),10.,20.,
)

def test_flat_hub_radius_has_exact_signed_zero_gap():
    sample=sample_spherical_tire_contact(
        Vec3(0.,.3,0.),Quaternion.identity(),0,FLAT,GEO,
    )
    assert sample.signed_surface_gap_m==pytest.approx(0.,abs=1e-12)
    assert sample.contact_world.y==pytest.approx(0.)
    assert sample.hub_world.y==pytest.approx(.3)


def test_radius_moves_contact_without_reinterpreting_hub_location():
    original=sample_spherical_tire_contact(
        Vec3(0.,.3,0.),Quaternion.identity(),0,FLAT,GEO,
    )
    larger=sample_spherical_tire_contact(
        Vec3(0.,.3,0.),Quaternion.identity(),0,FLAT,
        replace(GEO,rolling_radius_m=.32),
    )
    assert larger.hub_world==original.hub_world
    assert larger.signed_surface_gap_m==pytest.approx(-.02,abs=1e-12)


def test_inclined_plane_uses_true_3d_normal_not_world_vertical():
    from math import sin,cos
    angle=.1
    ramp=PlanarSurfacePatch(
        "ramp",Vec3(0.,0.,0.),
        Vec3(0.,sin(angle),cos(angle)),
        Vec3(0.,cos(angle),-sin(angle)),10.,20.,
    )
    point=sample_spherical_tire_contact(
        Vec3(0.,1.,0.),Quaternion.identity(),0,ramp,GEO,
    )
    assert (point.contact_world-point.hub_world).magnitude==pytest.approx(.3)
    assert point.normal==ramp.normal


def test_no_measured_constants_may_claim_calibration_gate_passed():
    empty=PhysicsCalibrationRegister((),"independent-verifier-pending")
    assert empty.missing(REQUIRED_WP7_MEASUREMENTS)==REQUIRED_WP7_MEASUREMENTS


def test_traceable_measurement_removes_only_its_own_missing_key():
    row=MeasuredPhysicsParameter(
        "mass_kg",1200.,"kg",1.,"mass-rig-log-01","calibrated force plate",
    )
    register=PhysicsCalibrationRegister((row,),"reviewer-signoff-reference")
    assert register.missing(REQUIRED_WP7_MEASUREMENTS)==REQUIRED_WP7_MEASUREMENTS[1:]


def test_nontraceable_and_duplicate_evidence_refused():
    with pytest.raises(ValueError):
        MeasuredPhysicsParameter("mass_kg",1200.,"kg",1.,"","scale")
    row=MeasuredPhysicsParameter("mass_kg",1200.,"kg",1.,"x","scale")
    with pytest.raises(ValueError):
        PhysicsCalibrationRegister((row,row),"reviewer")


@pytest.mark.parametrize("key,value",[
    ("rolling_radius_m",0.),("nominal_width_m",True),
    ("rolling_radius_m",1.1),
])
def test_invalid_wheel_geometry_refused(key,value):
    with pytest.raises(ValueError):
        replace(GEO,**{key:value})
