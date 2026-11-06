"""Unilateral contact onset and passive partial wheel loads at public solver seams."""
from dataclasses import replace
import pytest

from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.spatial_contact_dynamics import integrate_spatial_contact,SpatialContactPolicy
from f1rl.sandbox.coupled_longitudinal_dynamics import LongitudinalContactPolicy
from f1rl.sandbox.vehicle import VehicleControl
from tests.sandbox.test_road_relative_suspension import ramp_translation


@pytest.mark.parametrize('height,pitch,roll,active',[(.106,-.004,.01,1),(.06,-.03,0.,2),(.0745,-.02,.02,3)])
def test_uncompressed_springs_do_not_generate_force_during_partial_load_support(height,pitch,roll,active):
    f,_,_,s=ramp_translation()
    s=replace(s,force_model='road_relative_unilateral_v3')
    initial=replace(f.initial_state.vehicle,position=f.initial_state.vehicle.position+Vec3(0.,height,0.),
                    velocity=Vec3(.2,-1.,.5),orientation=Quaternion.from_axis_angle(Vec3(1.,0.,0.),pitch)
                    *Quaternion.from_axis_angle(Vec3(0.,0.,1.),roll))
    r=integrate_spatial_contact(initial,s,f.physics_config.body_geometry,f.track.surfaces,f.vehicle_parameters,
        VehicleControl(.2,.3,0.),SpatialContactPolicy(LongitudinalContactPolicy(.8,100000.),1800.,12000.),dt=.001)
    first=r.audit.substeps[0].contact_stages[0]
    assert sum(c.normal_force_n>0. for c in first)==active
    assert all(c.longitudinal_force_n==c.lateral_force_n==0. for c in first if c.normal_force_n==0.)
    if active==2:
        # Worked flat-road geometry gives rear gap .015231732821762795 m,
        # elastic 2486.0480153471162 N and HC factor 1+2000/2943 at -1 m/s.
        assert [c.normal_force_n for c in first[2:]]==pytest.approx([4175.513197370301]*2,abs=1e-8)
    assert all(s.dissipation_j>=0. and abs(s.energy_residual_j)<=s.energy_tolerance_j for s in r.audit.substeps)
    assert r.audit.max_linear_impulse_residual_ns<1e-7
    assert r.audit.max_angular_impulse_residual_nms<1e-7


def test_legacy_v2_force_law_is_preserved_as_an_explicit_different_contract():
    f,_,_,s=ramp_translation()
    initial=replace(f.initial_state.vehicle,position=f.initial_state.vehicle.position+Vec3(0.,.06,0.),
                    velocity=Vec3(0.,-1.,0.),orientation=Quaternion.from_axis_angle(Vec3(1.,0.,0.),-.03))
    r=integrate_spatial_contact(initial,s,f.physics_config.body_geometry,f.track.surfaces,f.vehicle_parameters,
        VehicleControl(0.,0.,0.),SpatialContactPolicy(LongitudinalContactPolicy(.8,100000.),1800.,12000.),dt=.001)
    assert all(c.normal_force_n>0. for c in r.audit.substeps[0].contact_stages[0])


def test_zero_damping_and_actuation_preserve_mechanical_energy_with_partial_loads():
    f,_,_,s=ramp_translation()
    s=replace(s,force_model='road_relative_unilateral_v3',damper_nspm=0.,angular_damping_nmsprad=0.)
    initial=replace(f.initial_state.vehicle,position=f.initial_state.vehicle.position+Vec3(0.,.06,0.),
                    velocity=Vec3(0.,-1.,0.),orientation=Quaternion.from_axis_angle(Vec3(1.,0.,0.),-.03))
    r=integrate_spatial_contact(initial,s,f.physics_config.body_geometry,f.track.surfaces,
        replace(f.vehicle_parameters,rolling_drag=0.),VehicleControl(0.,0.,0.),
        SpatialContactPolicy(LongitudinalContactPolicy(.8,100000.),1800.,0.),dt=.001)
    assert r.audit.dissipation_j==0. and r.audit.actuator_work_j==0.
    assert abs(r.audit.energy_residual_j)<1e-7


def test_unilateral_policy_never_falls_back_to_prescribed_path_integration():
    from f1rl.sandbox.independent_suspension_dynamics import integrate_independent_suspension,SuspensionResolutionError
    f,vehicle,dynamic,s=ramp_translation()
    s=replace(s,force_model='road_relative_unilateral_v3')
    with pytest.raises(SuspensionResolutionError,match='requires spatial'):
        integrate_independent_suspension(vehicle,vehicle,dynamic,s,f.physics_config.body_geometry,f.track.surfaces,
                                        dt=.001,gravity_mps2=f.vehicle_parameters.gravity)
    with pytest.raises(ValueError,match='force model/rate mapping'):
        replace(s,angular_rate_mapping='small_angle_v1')


def test_complete_unloading_is_rejected_until_an_airborne_solver_is_selected():
    from f1rl.sandbox.independent_suspension_dynamics import SuspensionResolutionError
    f,_,_,s=ramp_translation()
    s=replace(s,force_model='road_relative_unilateral_v3')
    initial=replace(f.initial_state.vehicle,velocity=Vec3(0.,2.,0.))
    with pytest.raises(SuspensionResolutionError,match='unloaded.*airborne'):
        integrate_spatial_contact(initial,s,f.physics_config.body_geometry,f.track.surfaces,f.vehicle_parameters,
            VehicleControl(0.,0.,0.),SpatialContactPolicy(LongitudinalContactPolicy(.8,100000.),1800.,0.),dt=.001)
