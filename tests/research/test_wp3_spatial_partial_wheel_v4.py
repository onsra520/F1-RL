"""Independent opt-in v4 partial wheel contact, no official Eval implication."""
from dataclasses import replace

import pytest

from src.simulation.coupled_longitudinal_dynamics import LongitudinalContactPolicy
from src.simulation.geometry.vector import Vec3
from src.simulation.independent_suspension_dynamics import SuspensionResolutionError
from src.simulation.multi_surface_support import _atlas
from src.simulation.spatial_contact_dynamics import (
    SpatialContactPolicy, integrate_spatial_road_edge_lifecycle,
)
from src.simulation.spatial_partial_wheel_contact import sample_spatial_partial_wheels
from src.simulation.vehicle import VehicleControl
from tests.sandbox.test_partial_wheel_release import CONTACT,PHYSICS,road,pose,BODY

PLAN=(road(),)
ATLAS=_atlas(PLAN,CONTACT.seam_height_tolerance_m)


@pytest.mark.parametrize(("z","expected"),[
    (0.,4),(1.3,2),(2.9,0),(-1.3,2),
])
def test_support_count_at_authored_platform_edges(z,expected):
    sample=sample_spatial_partial_wheels(pose(z),CONTACT,ATLAS)
    assert sample.support_count==expected
    assert sum(x=="outside_road" for x in sample.classification)==4-expected


def test_road_half_width_leaves_two_wheels_supported():
    from src.simulation.track.patch import PlanarSurfacePatch
    narrow=road(width=1.,length=20.)
    atlas=_atlas((narrow,),CONTACT.seam_height_tolerance_m)
    s=sample_spatial_partial_wheels(
        replace(pose(0.),position=Vec3(.3,.3,0.)),
        CONTACT,atlas)
    assert s.support_count==2


def test_single_wheel_can_be_supported_on_asymmetric_edge():
    from src.simulation.track.patch import PlanarSurfacePatch
    patch=PlanarSurfacePatch("one",Vec3(-.4,0.,.8),Vec3(0,0,1),Vec3(0,1,0),.15,.15)
    atlas=_atlas((patch,),CONTACT.seam_height_tolerance_m)
    assert sample_spatial_partial_wheels(pose(),CONTACT,atlas).support_count==1


def test_three_wheel_case_with_one_missing_footprint():
    from src.simulation.track.patch import PlanarSurfacePatch
    # Keep front-right, rear-left, rear-right; front-left outside.
    a=PlanarSurfacePatch("left-rear",Vec3(-.4,0.,-.8),Vec3(0,0,1),Vec3(0,1,0),.4,.4)
    b=PlanarSurfacePatch("right",Vec3(.4,0.,0.),Vec3(0,0,1),Vec3(0,1,0),.4,2.3)
    atlas=_atlas((a,b),CONTACT.seam_height_tolerance_m)
    assert sample_spatial_partial_wheels(pose(),CONTACT,atlas).support_count==3


def test_travel_over_bump_rejected_not_released():
    state=replace(pose(),position=Vec3(0.,.1,0.))
    with pytest.raises(SuspensionResolutionError,match="bump"):
        sample_spatial_partial_wheels(state,CONTACT,ATLAS)


def test_wheel_above_droop_is_unloaded_not_false_edge_departure():
    state=replace(pose(),position=Vec3(0.,2.,0.))
    check=sample_spatial_partial_wheels(state,CONTACT,ATLAS)
    assert check.support_count==0
    assert check.classification==("unloaded_above_droop",)*4


def spatial_policy():
    return SpatialContactPolicy(LongitudinalContactPolicy(.8,100000.),1800.,12000.)


def test_optin_lifecycle_handles_safe_four_supported_reference():
    from tests.sandbox.test_road_relative_suspension import ramp_translation
    fac,_,_,s=ramp_translation()
    v=fac.initial_state.vehicle
    p=replace(s,force_model="road_relative_unilateral_v3")
    result=integrate_spatial_road_edge_lifecycle(
        v,p,fac.physics_config.body_geometry,fac.track.surfaces,
        fac.vehicle_parameters,VehicleControl(0.,0.,0.),spatial_policy(),dt=.000001)
    assert result.audit.version=="spatial_road_edge_partial_lifecycle_rk4_v4"
    assert result.audit.max_linear_impulse_residual_ns<1e-7
    assert result.audit.max_angular_impulse_residual_nms<1e-7
    assert len(result.audit.substeps)>0


def test_new_mode_cannot_modify_original_supported_v3_path():
    from src.simulation.spatial_contact_dynamics import integrate_spatial_lifecycle
    from tests.sandbox.test_road_relative_suspension import ramp_translation
    fac,_,_,s=ramp_translation()
    p=replace(s,force_model="road_relative_unilateral_v3")
    v=fac.initial_state.vehicle
    a=integrate_spatial_lifecycle(
        v,p,fac.physics_config.body_geometry,fac.track.surfaces,
        fac.vehicle_parameters,VehicleControl(0.,0.,0.),spatial_policy(),
        dt=.000001,phase_domain="atlas_rotating_v3")
    assert a.audit.version=="spatial_rotating_compliant_lifecycle_rk4_v3"


@pytest.mark.parametrize(("z","expected"),[(0.,4),(1.3,2)])
def test_optin_6dof_spatial_rk4_uses_only_witnessed_wheel_loads(z,expected):
    from tests.sandbox.test_road_relative_suspension import ramp_translation
    fac,_,_,_=ramp_translation()
    v=pose(z)
    p=replace(PHYSICS,force_model="road_relative_unilateral_v3",
              angular_rate_mapping="yaw_rx_rz_v1")
    result=integrate_spatial_road_edge_lifecycle(
        v,p,BODY,PLAN,fac.vehicle_parameters,
        VehicleControl(0.,0.,0.),spatial_policy(),dt=1e-8)
    leaf=result.audit.substeps[0]
    for stage in leaf.contact_stages:
        assert len(stage)==expected
        assert all(c.normal_force_n>=0 for c in stage)
    assert abs(leaf.energy_residual_j)<=leaf.energy_tolerance_j
    assert leaf.linear_impulse_residual_ns<1e-7
    assert leaf.angular_impulse_residual_nms<1e-7


def test_optin_road_edge_rejects_penetrating_body_even_with_support_present():
    from tests.sandbox.test_road_relative_suspension import ramp_translation
    fac,_,_,_=ramp_translation()
    p=replace(PHYSICS,force_model="road_relative_unilateral_v3",
              angular_rate_mapping="yaw_rx_rz_v1")
    v=replace(pose(1.3),position=Vec3(0.,.1,1.3))
    with pytest.raises(SuspensionResolutionError):
        integrate_spatial_road_edge_lifecycle(
            v,p,BODY,PLAN,fac.vehicle_parameters,
            VehicleControl(0.,0.,0.),spatial_policy(),dt=1e-8)


def test_finite_tire_edge_release_conserves_work_without_hidden_energy():
    from tests.sandbox.test_road_relative_suspension import ramp_translation
    f,_,_,_=ramp_translation()
    v=pose(1.199,speed=4.)
    physics=replace(PHYSICS,force_model="road_relative_unilateral_v3",
                    angular_rate_mapping="yaw_rx_rz_v1")
    result=integrate_spatial_road_edge_lifecycle(
        v,physics,BODY,PLAN,f.vehicle_parameters,
        VehicleControl(0.,0.,0.),spatial_policy(),dt=.001)
    dissipated=sum(leaf.road_edge_release_dissipation_j for leaf in result.audit.substeps)
    # v4's finite tyre footprint now uses the conservative coverage-gradient
    # force for edge spring energy. No artificial one-step edge dissipation
    # is needed or licensed in this continuous-contact test.
    assert dissipated==0.
    assert result.audit.dissipation_j>=dissipated
    for leaf in result.audit.substeps:
        assert abs(leaf.energy_residual_j)<=leaf.energy_tolerance_j
    assert result.audit.max_linear_impulse_residual_ns<1e-7
    assert result.audit.max_angular_impulse_residual_nms<1e-7
