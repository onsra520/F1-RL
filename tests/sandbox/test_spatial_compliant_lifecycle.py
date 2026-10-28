"""Compliant onset/release, ballistic intervals and bounded phase evidence."""
from dataclasses import replace
import pytest
from src.sandbox.geometry.vector import Vec3
from src.sandbox.vehicle import VehicleControl
from src.sandbox.spatial_contact_dynamics import integrate_spatial_lifecycle, SpatialContactPolicy
from src.sandbox.coupled_longitudinal_dynamics import LongitudinalContactPolicy
from tests.sandbox.test_road_relative_suspension import ramp_translation


def test_actual_release_and_landing_have_bounded_phase_brackets_without_impulse():
    f,_,_,s=ramp_translation()
    s=replace(s,force_model='road_relative_unilateral_v3')
    state=replace(f.initial_state.vehicle,velocity=Vec3(0.,2.,0.))
    policy=SpatialContactPolicy(LongitudinalContactPolicy(.8,100000.),1800.,12000.)
    phases=[]
    initial_energy=None
    for tick in range(120):
        r=integrate_spatial_lifecycle(state,s,f.physics_config.body_geometry,f.track.surfaces,
            f.vehicle_parameters,VehicleControl(0.,0.,0.),policy,dt=.01)
        if initial_energy is None:
            initial_energy=r.audit.substeps[0].initial_energy_j
        for bracket in r.audit.phase_brackets:
            assert 0. <= bracket.lower_time_s <= bracket.upper_time_s <= .01+1e-12
            assert bracket.upper_time_s-bracket.lower_time_s <= 1e-8
            phases.append(bracket.kind)
        assert all(a.dissipation_j>=0. and abs(a.energy_residual_j)<=a.energy_tolerance_j for a in r.audit.substeps)
        if not state.grounded and not r.suspension.vehicle.grounded:
            assert r.suspension.vehicle.velocity.y==pytest.approx(state.velocity.y-f.vehicle_parameters.gravity*.01,abs=1e-9)
            assert r.audit.actuator_work_j==r.audit.dissipation_j==0.
            assert all(c.initial_velocity is not None for c in r.center_segments)
        state=r.suspension.vehicle
    assert phases[:2]==['takeoff','landing']
    assert state.grounded and abs(state.velocity.y)<.03
    assert r.audit.substeps[-1].final_energy_j <= initial_energy+1e-5


@pytest.mark.parametrize('phase_domain',['horizontal_v1','atlas_gradient_v2','atlas_rotating_v3'])
def test_a_narrow_continuous_ridge_cannot_hide_contact_between_rk_stages(phase_domain):
    from math import hypot
    from src.sandbox.track.patch import PlanarSurfacePatch
    from src.sandbox.independent_suspension_dynamics import SuspensionResolutionError
    f,_,_,s=ramp_translation()
    s=replace(s,force_model='road_relative_unilateral_v3',integration_substeps=1)
    vertices=[(-10.,0.),(5.58,0.),(5.6,.1),(5.62,0.),(20.,0.)]
    patches=[]
    for i,((z0,h0),(z1,h1)) in enumerate(zip(vertices,vertices[1:])):
        dz,dh=z1-z0,h1-h0
        patches.append(PlanarSurfacePatch(str(i),Vec3(0.,(h0+h1)/2.,(z0+z1)/2.),
            Vec3(0.,dh,dz).normalized(),Vec3(0.,dz,-dh).normalized(),20.,hypot(dz,dh)))
    state=replace(f.initial_state.vehicle,position=Vec3(0.,.68,4.),velocity=Vec3(0.,0.,4.),grounded=False)
    policy=SpatialContactPolicy(LongitudinalContactPolicy(.8,100000.),1800.,12000.)
    # This ridge produces contact, then angular departure outside the fixed-air
    # attitude domain. It previously passed as one force-free ballistic leaf.
    with pytest.raises(SuspensionResolutionError):
        integrate_spatial_lifecycle(state,s,f.physics_config.body_geometry,tuple(patches),
            f.vehicle_parameters,VehicleControl(0.,0.,0.),policy,dt=.1,phase_domain=phase_domain)


def test_airborne_actions_and_ground_modifiers_cannot_create_propulsion_or_drag():
    from src.sandbox.dynamics.modifiers import MotionModifiers
    f,_,_,s=ramp_translation()
    s=replace(s,force_model='road_relative_unilateral_v3')
    state=replace(f.initial_state.vehicle,position=Vec3(0.,5.,4.),velocity=Vec3(0.,0.,4.),grounded=False)
    policy=SpatialContactPolicy(LongitudinalContactPolicy(.8,100000.),1800.,12000.)
    args=(state,s,f.physics_config.body_geometry,f.track.surfaces,f.vehicle_parameters)
    reference=integrate_spatial_lifecycle(*args,VehicleControl(0.,0.,0.),policy,dt=.1)
    acted=integrate_spatial_lifecycle(*args,VehicleControl(.9,.8,.7),policy,dt=.1,
        modifiers=MotionModifiers(longitudinal_acceleration_bonus=10.,rolling_drag_multiplier=3.))
    assert acted==reference
    assert acted.audit.actuator_work_j==acted.audit.dissipation_j==0.


@pytest.mark.parametrize('case',['rotating','phase','droop','edge'])
def test_unsupported_lifecycle_domains_reject_instead_of_falling_back(case):
    from src.sandbox.independent_suspension_dynamics import SuspensionResolutionError
    f,_,_,s=ramp_translation()
    s=replace(s,force_model='road_relative_unilateral_v3')
    state=replace(f.initial_state.vehicle,position=Vec3(0.,5.,4.),velocity=Vec3(0.,0.,4.),grounded=False)
    if case=='rotating':
        state=replace(state,angular_velocity=Vec3(0.,.1,0.))
    elif case=='phase':
        state=replace(state,grounded=True)
    elif case=='droop':
        s=replace(s,contact=replace(s.contact,max_droop_m=.05))
    else:
        state=replace(state,position=Vec3(100.,5.,4.))
    with pytest.raises(SuspensionResolutionError):
        integrate_spatial_lifecycle(state,s,f.physics_config.body_geometry,f.track.surfaces,
            f.vehicle_parameters,VehicleControl(0.,0.,0.),
            SpatialContactPolicy(LongitudinalContactPolicy(.8,100000.),1800.,12000.),dt=.1)


@pytest.mark.parametrize('phase_domain',['horizontal_v1','atlas_gradient_v2','atlas_rotating_v3'])
def test_simultaneous_narrow_road_valleys_cannot_hide_a_ground_to_air_transition(phase_domain):
    from math import hypot
    from src.sandbox.track.patch import PlanarSurfacePatch
    from src.sandbox.independent_suspension_dynamics import SuspensionResolutionError
    f,_,_,s=ramp_translation()
    s=replace(s,force_model='road_relative_unilateral_v3',integration_substeps=1)
    vertices=[(-10.,0.),(2.58,0.),(2.6,-.15),(2.62,0.),(5.58,0.),(5.6,-.15),(5.62,0.),(20.,0.)]
    patches=[]
    for i,((z0,h0),(z1,h1)) in enumerate(zip(vertices,vertices[1:])):
        dz,dh=z1-z0,h1-h0
        patches.append(PlanarSurfacePatch(str(i),Vec3(0.,(h0+h1)/2.,(z0+z1)/2.),
            Vec3(0.,dh,dz).normalized(),Vec3(0.,dz,-dh).normalized(),20.,hypot(dz,dh)))
    state=replace(f.initial_state.vehicle,position=Vec3(0.,.5,4.),velocity=Vec3(0.,0.,4.))
    try:
        r=integrate_spatial_lifecycle(state,s,f.physics_config.body_geometry,tuple(patches),
            replace(f.vehicle_parameters,rolling_drag=0.),VehicleControl(0.,0.,0.),
            SpatialContactPolicy(LongitudinalContactPolicy(.8,100000.),1800.,12000.),dt=.1,phase_domain=phase_domain)
    except SuspensionResolutionError:
        return  # Unproved or rotating recontact must reject, never manufacture support.
    assert any(b.kind=='takeoff' for b in r.audit.phase_brackets)
