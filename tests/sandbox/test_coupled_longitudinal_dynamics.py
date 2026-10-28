"""Force/work oracles for coupled planar contact; never calibrated physics."""
from dataclasses import replace
from math import cos, sin

import pytest

from src.simulation.coupled_longitudinal_dynamics import (
    LongitudinalContactPolicy, integrate_coupled_longitudinal,
)
from src.simulation.geometry.vector import Vec3
from src.simulation.vehicle import VehicleControl
from tests.sandbox.test_road_relative_suspension import ramp_translation


def test_flat_equilibrium_preserves_all_momentum_and_energy():
    factory, _, _, suspension = ramp_translation()
    initial = factory.initial_state
    policy = LongitudinalContactPolicy(traction_mu=.8, max_drive_power_w=100000.)
    result = integrate_coupled_longitudinal(initial.vehicle, initial.suspension,
        suspension, factory.physics_config.body_geometry, factory.track.surfaces,
        factory.vehicle_parameters, VehicleControl(0., 0., 0.), policy,
        dt=factory.simulation_config.fixed_dt)
    assert (result.suspension.vehicle.position-initial.vehicle.position).magnitude < 1e-12
    assert result.suspension.vehicle.velocity.magnitude < 1e-12
    assert result.suspension.vehicle.angular_velocity.magnitude < 1e-12
    assert result.audit.actuator_work_j == pytest.approx(0., abs=1e-12)
    assert result.audit.dissipation_j == pytest.approx(0., abs=1e-12)
    assert abs(result.audit.energy_residual_j) < 1e-10
    assert result.audit.max_linear_impulse_residual_ns < 1e-9
    assert result.audit.max_angular_impulse_residual_nms < 1e-9


def test_frictionless_incline_matches_exact_gravity_slide():
    factory, vehicle, dynamic, suspension = ramp_translation()
    angle = -dynamic.pitch_x_rad
    gravity = factory.vehicle_parameters.gravity
    # At normal equilibrium N=mg*cos(angle). Uniform vertical unloading
    # leaves each spring's compression constant throughout the exact slide.
    lift = suspension.mass_kg*gravity*sin(angle)**2/(4.*suspension.spring_stiffness_npm)
    vehicle = replace(vehicle, position=vehicle.position+Vec3(0.,lift,0.), velocity=Vec3(0.,0.,0.))
    dynamic = replace(dynamic, height_m=vehicle.position.y, heave_velocity_mps=0.)
    suspension = replace(suspension, damper_nspm=0., angular_damping_nmsprad=0.)
    parameters = replace(factory.vehicle_parameters, rolling_drag=0.)
    dt = .1
    result = integrate_coupled_longitudinal(vehicle, dynamic, suspension,
        factory.physics_config.body_geometry, factory.track.surfaces, parameters,
        VehicleControl(0.,0.,0.), LongitudinalContactPolicy(0.,100000.), dt=dt)
    acceleration = Vec3(0.,-gravity*sin(angle)**2,-gravity*sin(angle)*cos(angle))
    assert (result.suspension.vehicle.velocity-acceleration*dt).magnitude < 1e-10
    assert (result.suspension.vehicle.position-vehicle.position-acceleration*(.5*dt*dt)).magnitude < 1e-10
    assert result.suspension.dynamics.pitch_x_rad == pytest.approx(-angle,abs=1e-12)
    assert abs(result.suspension.dynamics.pitch_rate_radps) < 1e-11
    assert result.audit.actuator_work_j == 0.
    assert result.audit.dissipation_j == 0.
    assert abs(result.audit.energy_residual_j) < 1e-9


@pytest.mark.parametrize('mu,throttle,expected_acceleration', [
    (0.,1.,0.), (.01,1.,.0981), (1.,.5,None),
])
def test_flat_initial_traction_and_pitch_obey_newton_euler(mu,throttle,expected_acceleration):
    factory, _, _, suspension = ramp_translation()
    initial = factory.initial_state
    expected = factory.vehicle_parameters.max_acceleration*throttle if expected_acceleration is None else expected_acceleration
    dt = 1e-6
    result = integrate_coupled_longitudinal(initial.vehicle,initial.suspension,suspension,
        factory.physics_config.body_geometry,factory.track.surfaces,factory.vehicle_parameters,
        VehicleControl(0.,throttle,0.),LongitudinalContactPolicy(mu,100000.),dt=dt)
    # All four contact reference points start 0.5 m below the mass centre.
    pitch_acceleration = -.5*suspension.mass_kg*expected/suspension.pitch_inertia_kg_m2
    assert result.suspension.vehicle.velocity.z/dt == pytest.approx(expected,abs=2e-5)
    assert result.suspension.vehicle.angular_velocity.x/dt == pytest.approx(pitch_acceleration,abs=2e-5)
    for leaf in result.audit.substeps:
        for stage in leaf.contact_stages:
            assert all(abs(c.tangent_force_n)<=mu*c.normal_force_n+1e-10 for c in stage)


@pytest.mark.parametrize('brake,power', [(0.,100.),(1.,100000.)])
def test_actuator_power_cap_and_dissipative_braking(brake,power):
    factory, _, _, suspension = ramp_translation()
    initial = factory.initial_state
    vehicle = replace(initial.vehicle,velocity=Vec3(0.,0.,3.))
    result = integrate_coupled_longitudinal(vehicle,initial.suspension,suspension,
        factory.physics_config.body_geometry,factory.track.surfaces,factory.vehicle_parameters,
        VehicleControl(0.,1.-brake,brake),LongitudinalContactPolicy(.8,power),dt=.04)
    for leaf in result.audit.substeps:
        assert abs(leaf.energy_residual_j)<=leaf.energy_tolerance_j
        for stage in leaf.contact_stages:
            assert sum(max(0.,c.tangent_force_n*c.tangent_speed_mps) for c in stage)<=power+1e-8
            if brake:
                assert all(c.tangent_force_n*c.tangent_speed_mps<=0. for c in stage)
    assert result.audit.max_linear_impulse_residual_ns < 1e-8
    assert result.audit.max_angular_impulse_residual_nms < 1e-8
    if brake:
        assert result.audit.actuator_work_j < 0.
        assert result.suspension.vehicle.speed < vehicle.speed
    else:
        assert result.audit.actuator_work_j == pytest.approx(power*.04,abs=1e-9)


def test_force_jump_at_actual_incline_seam_refines_without_relaxing_work_tolerance():
    from src.simulation.geometry.rotation import Quaternion
    from src.simulation.independent_suspension_dynamics import SuspensionDynamicsState, SuspensionResolutionError

    factory, _, _, suspension = ramp_translation()
    dynamic = SuspensionDynamicsState(.49999747118574095,0.,-.0031804505282522146)
    vehicle = replace(factory.initial_state.vehicle,position=Vec3(0.,dynamic.height_m,48.4984),
        velocity=Vec3(0.,0.,3.5),orientation=Quaternion.from_axis_angle(Vec3(1.,0.,0.),dynamic.pitch_x_rad))
    suspension = replace(suspension,integration_substeps=1)
    args = (vehicle,dynamic,suspension,factory.physics_config.body_geometry,
            factory.track.surfaces,factory.vehicle_parameters,VehicleControl(0.,0.,0.))
    with pytest.raises(SuspensionResolutionError,match='refinement limit'):
        integrate_coupled_longitudinal(*args,LongitudinalContactPolicy(.8,100000.,max_refinements=8),dt=.001)
    result = integrate_coupled_longitudinal(*args,LongitudinalContactPolicy(.8,100000.),dt=.001)
    assert len(result.audit.substeps)>1
    assert len(result.audit.substeps)<100
    assert result.center_segments[-1].end_time_s == pytest.approx(.001,abs=1e-15)
    assert any(c.patch_id=='road-incline' for s in result.audit.substeps for stage in s.contact_stages for c in stage)
    assert all(abs(s.energy_residual_j)<=s.energy_tolerance_j for s in result.audit.substeps)


def test_smooth_force_trajectory_converges_when_substeps_are_halved():
    factory, _, _, suspension = ramp_translation()
    initial = factory.initial_state
    results = []
    for count in (1,2,4,128):
        result = integrate_coupled_longitudinal(initial.vehicle,initial.suspension,
            replace(suspension,integration_substeps=count),factory.physics_config.body_geometry,
            factory.track.surfaces,factory.vehicle_parameters,VehicleControl(0.,.8,0.),
            LongitudinalContactPolicy(.8,100000.,energy_relative_tolerance=1e-5,max_refinements=0),dt=.01)
        results.append(result.suspension.vehicle)
        assert len(result.audit.substeps)==count
    errors = [(r.velocity-results[-1].velocity).magnitude
              +(r.angular_velocity-results[-1].angular_velocity).magnitude for r in results[:-1]]
    assert errors[0]>8.*errors[1]>64.*errors[2]>0.


def test_energy_tolerance_does_not_depend_on_world_height_origin():
    factory, _, _, suspension = ramp_translation()
    initial = factory.initial_state
    results = []
    for offset in (0.,100.):
        shift = Vec3(0.,offset,0.)
        result = integrate_coupled_longitudinal(
            replace(initial.vehicle,position=initial.vehicle.position+shift),
            replace(initial.suspension,height_m=initial.suspension.height_m+offset),suspension,
            factory.physics_config.body_geometry,tuple(replace(p,origin=p.origin+shift) for p in factory.track.surfaces),
            factory.vehicle_parameters,VehicleControl(0.,.5,0.),LongitudinalContactPolicy(.8,100000.),dt=.02)
        results.append(result)
    assert (results[0].suspension.vehicle.velocity-results[1].suspension.vehicle.velocity).magnitude<1e-10
    assert [s.energy_tolerance_j for s in results[0].audit.substeps] == pytest.approx(
        [s.energy_tolerance_j for s in results[1].audit.substeps],rel=1e-10)


@pytest.mark.parametrize('field,value', [
    ('traction_mu',float('nan')),('traction_mu',-1.),('max_drive_power_w',0.),
    ('brake_speed_scale_mps',0.),('energy_relative_tolerance',1e-4),
    ('energy_absolute_tolerance_j',float('inf')),('max_refinements',25),('max_refinements',True),
])
def test_unregistered_contact_policy_is_rejected(field,value):
    with pytest.raises(ValueError):
        replace(LongitudinalContactPolicy(.8,100000.),**{field:value})


@pytest.mark.parametrize('change', ['lateral','yaw','yaw_orientation','roll','airborne','inconsistent_heave','speed','travel'])
def test_kernel_rejects_unsupported_or_inconsistent_physical_state(change):
    from src.simulation.independent_suspension_dynamics import SuspensionResolutionError
    factory, _, _, suspension = ramp_translation()
    vehicle,dynamic = factory.initial_state.vehicle,factory.initial_state.suspension
    if change=='lateral': vehicle=replace(vehicle,velocity=Vec3(.1,0.,0.))
    if change=='yaw': vehicle=replace(vehicle,angular_velocity=Vec3(0.,.1,0.))
    if change=='yaw_orientation':
        from src.simulation.geometry.rotation import Quaternion
        vehicle=replace(vehicle,orientation=Quaternion.from_axis_angle(Vec3(0.,1.,0.),2e-5))
    if change=='roll': dynamic=replace(dynamic,roll_z_rad=.01)
    if change=='airborne': vehicle=replace(vehicle,grounded=False)
    if change=='inconsistent_heave': dynamic=replace(dynamic,heave_velocity_mps=.1)
    if change=='speed': vehicle=replace(vehicle,velocity=Vec3(0.,0.,100.))
    if change=='travel':
        vehicle=replace(vehicle,position=vehicle.position+Vec3(0.,1.,0.))
        dynamic=replace(dynamic,height_m=vehicle.position.y)
    with pytest.raises(SuspensionResolutionError):
        integrate_coupled_longitudinal(vehicle,dynamic,suspension,
            factory.physics_config.body_geometry,factory.track.surfaces,factory.vehicle_parameters,
            VehicleControl(0.,0.,0.),LongitudinalContactPolicy(.8,100000.),dt=.1)
