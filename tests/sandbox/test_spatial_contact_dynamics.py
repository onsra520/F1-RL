"""Newton/Euler and energy oracles for supported spatial contact."""
from dataclasses import replace
from math import cos, sin, tan

import pytest

from src.sandbox.spatial_contact_dynamics import SpatialContactPolicy, integrate_spatial_contact
from src.sandbox.coupled_longitudinal_dynamics import LongitudinalContactPolicy, integrate_coupled_longitudinal
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.geometry.vector import Vec3
from src.sandbox.vehicle import VehicleControl
from tests.sandbox.test_road_relative_suspension import ramp_translation


def run(vehicle, factory, suspension, control, *, dt=.01, policy=None):
    return integrate_spatial_contact(vehicle,suspension,factory.physics_config.body_geometry,
        factory.track.surfaces,factory.vehicle_parameters,control,
        policy or SpatialContactPolicy(LongitudinalContactPolicy(.8,100000.),1800.,12000.),dt=dt)


def test_flat_equilibrium_has_no_invented_momentum_or_work():
    f,_,_,s=ramp_translation()
    initial=f.initial_state.vehicle
    r=run(initial,f,s,VehicleControl(0.,0.,0.))
    assert (r.suspension.vehicle.position-initial.position).magnitude<1e-12
    assert r.suspension.vehicle.velocity.magnitude<1e-12
    assert r.suspension.vehicle.angular_velocity.magnitude<1e-12
    assert abs(r.audit.energy_residual_j)<1e-10
    assert r.audit.actuator_work_j==0. and r.audit.dissipation_j==0.


def test_zero_lateral_planar_solution_matches_existing_force_coupled_kernel():
    f,vehicle,dynamic,s=ramp_translation()
    control=VehicleControl(0.,.5,0.)
    p=LongitudinalContactPolicy(.8,100000.)
    expected=integrate_coupled_longitudinal(vehicle,dynamic,s,f.physics_config.body_geometry,
        f.track.surfaces,f.vehicle_parameters,control,p,dt=.01)
    actual=run(vehicle,f,s,control)
    a,b=actual.suspension.vehicle,expected.suspension.vehicle
    assert (a.position-b.position).magnitude<1e-10
    assert (a.velocity-b.velocity).magnitude<1e-10
    assert (a.angular_velocity-b.angular_velocity).magnitude<1e-9
    assert actual.audit.actuator_work_j==pytest.approx(expected.audit.actuator_work_j,abs=1e-7)


def test_front_steering_force_obeys_full_initial_newton_euler_acceleration():
    f,_,_,s=ramp_translation()
    initial=f.initial_state.vehicle
    control=VehicleControl(.4,.5,0.)
    delta=.4*f.vehicle_parameters.max_steering_angle
    force=s.mass_kg*(.5*f.vehicle_parameters.max_acceleration)/4.
    expected_x=2.*force*sin(delta)/s.mass_kg
    expected_z=2.*force*(1.+cos(delta))/s.mass_kg
    # Front contacts have z=+1.5; only they receive a steered lateral drive force.
    expected_yaw=3.*force*sin(delta)/1800.
    actual=run(initial,f,s,control,dt=1e-7)
    v=actual.suspension.vehicle
    assert v.velocity.x/1e-7==pytest.approx(expected_x,abs=2e-5)
    assert v.velocity.z/1e-7==pytest.approx(expected_z,abs=2e-5)
    assert v.angular_velocity.y/1e-7==pytest.approx(expected_yaw,abs=2e-5)
    assert actual.audit.max_linear_impulse_residual_ns<1e-9
    assert actual.audit.max_angular_impulse_residual_nms<1e-9


@pytest.mark.parametrize('yaw',[-.7,.7,2.])
def test_flat_contact_is_covariant_under_world_yaw(yaw):
    f,_,_,s=ramp_translation()
    vehicle=replace(f.initial_state.vehicle,velocity=Vec3(.2,0.,2.),angular_velocity=Vec3(.01,.02,.03))
    rotation=Quaternion.from_axis_angle(Vec3(0.,1.,0.),yaw)
    rotated=replace(vehicle,velocity=rotation.rotate(vehicle.velocity),
                    angular_velocity=rotation.rotate(vehicle.angular_velocity),orientation=rotation*vehicle.orientation)
    control=VehicleControl(.3,.4,.1)
    first=run(vehicle,f,s,control,dt=.003).suspension.vehicle
    second=run(rotated,f,s,control,dt=.003).suspension.vehicle
    assert (second.velocity-rotation.rotate(first.velocity)).magnitude<1e-10
    assert (second.angular_velocity-rotation.rotate(first.angular_velocity)).magnitude<1e-10
    assert (second.position-vehicle.position-rotation.rotate(first.position-vehicle.position)).magnitude<1e-10


def test_friction_circle_power_and_lateral_dissipation_remain_bounded():
    f,_,_,s=ramp_translation()
    vehicle=replace(f.initial_state.vehicle,velocity=Vec3(.5,0.,3.))
    policy=SpatialContactPolicy(LongitudinalContactPolicy(.05,100.),1800.,12000.)
    r=run(vehicle,f,s,VehicleControl(.6,1.,0.),policy=policy)
    for leaf in r.audit.substeps:
        assert abs(leaf.energy_residual_j)<=leaf.energy_tolerance_j
        assert leaf.dissipation_j>=0.
        for stage in leaf.contact_stages:
            for c in stage:
                assert (c.longitudinal_force_n**2+c.lateral_force_n**2)**.5<=.05*c.normal_force_n+1e-9
                assert c.lateral_force_n*c.lateral_speed_mps<=1e-10
                assert c.longitudinal_force_n*c.longitudinal_speed_mps<=25.+1e-9


def test_two_continuous_patches_with_mixed_normals_have_actual_lateral_impulse():
    from src.sandbox.track.patch import PlanarSurfacePatch
    f,_,_,s=ramp_translation()
    angle=.05
    surfaces=(PlanarSurfacePatch('left',Vec3(-3.,0.,25.),Vec3(0.,0.,1.),Vec3(0.,1.,0.),6.,50.),
              PlanarSurfacePatch('right',Vec3(3.,3.*tan(angle),25.),Vec3(0.,0.,1.),
                                 Vec3(-sin(angle),cos(angle),0.),6./cos(angle),50.))
    height=.5+.375*tan(angle)
    vehicle=replace(f.initial_state.vehicle,position=Vec3(0.,height,4.))
    policy=SpatialContactPolicy(LongitudinalContactPolicy(0.,100000.),1800.,0.)
    dt=1e-7
    r=integrate_spatial_contact(vehicle,s,f.physics_config.body_geometry,surfaces,
        f.vehicle_parameters,VehicleControl(0.,0.,0.),policy,dt=dt)
    right_load=s.mass_kg*f.vehicle_parameters.gravity/4.+s.spring_stiffness_npm*.375*tan(angle)
    expected_acceleration=-2.*right_load*tan(angle)/s.mass_kg
    assert r.suspension.vehicle.velocity.x/dt==pytest.approx(expected_acceleration,abs=2e-5)
    assert {c.patch_id for c in r.audit.substeps[0].contact_stages[0]}=={'left','right'}
    assert r.audit.max_angular_impulse_residual_nms<1e-9


@pytest.mark.parametrize('change',['airborne','missing_support','overspeed'])
def test_spatial_kernel_rejects_states_outside_registered_domain(change):
    from src.sandbox.independent_suspension_dynamics import SuspensionResolutionError
    f,_,_,s=ramp_translation()
    vehicle=f.initial_state.vehicle
    if change=='airborne': vehicle=replace(vehicle,grounded=False)
    if change=='missing_support': vehicle=replace(vehicle,position=vehicle.position+Vec3(20.,0.,0.))
    if change=='overspeed': vehicle=replace(vehicle,velocity=Vec3(0.,0.,100.))
    with pytest.raises(SuspensionResolutionError) as error:
        run(vehicle,f,s,VehicleControl(0.,0.,0.))
    assert error.value.failure_code == "uncertified_physics"


def test_spatial_kernel_classifies_only_the_registered_outside_support_verdict(monkeypatch):
    from types import SimpleNamespace
    from src.sandbox import spatial_contact_dynamics
    from src.sandbox.independent_suspension_dynamics import SuspensionResolutionError

    f,_,_,s=ramp_translation()
    monkeypatch.setattr(spatial_contact_dynamics, "certify_multisurface_support",
                        lambda *_args, **_kwargs: SimpleNamespace(
                            valid=False, status="outside",
                            reason="invalid endpoint body or wheel support"))
    with pytest.raises(SuspensionResolutionError) as error:
        run(f.initial_state.vehicle,f,s,VehicleControl(0.,0.,0.))
    assert error.value.failure_code == "unsupported_contact"


def test_spatial_kernel_classifies_witnessed_swept_outside_as_unsupported_contact(monkeypatch):
    from types import SimpleNamespace
    from src.sandbox import spatial_contact_dynamics
    from src.sandbox.independent_suspension_dynamics import SuspensionResolutionError

    f,_,_,s=ramp_translation()
    monkeypatch.setattr(spatial_contact_dynamics, "certify_multisurface_support",
                        lambda *_args, **_kwargs: SimpleNamespace(
                            valid=False, status="outside",
                            reason="witnessed unsupported body or wheel pose within motion"))
    with pytest.raises(SuspensionResolutionError) as error:
        run(f.initial_state.vehicle,f,s,VehicleControl(0.,0.,0.))
    assert error.value.failure_code == "unsupported_contact"


def test_nonplanar_spin_acceleration_matches_independent_euler_equations():
    f,_,_,s=ramp_translation()
    q=Quaternion.from_axis_angle(Vec3(1.,0.,0.),.04)*Quaternion.from_axis_angle(Vec3(0.,0.,1.),.02)
    vehicle=replace(f.initial_state.vehicle,orientation=q,velocity=Vec3(.1,.03,1.),angular_velocity=Vec3(.05,.06,.04))
    s=replace(s,angular_damping_nmsprad=0.)
    f=replace(f,vehicle_parameters=replace(f.vehicle_parameters,rolling_drag=0.))
    force=Vec3(0.,-s.mass_kg*f.vehicle_parameters.gravity,0.)
    torque=Vec3(0.,0.,0.)
    for local in s.contact.wheels.positions:
        arm=q.rotate(local)
        height=vehicle.position.y+arm.y
        rate=(vehicle.velocity+vehicle.angular_velocity.cross(arm)).y
        load=max(0.,s.mass_kg*f.vehicle_parameters.gravity/4.-s.spring_stiffness_npm*height-s.damper_nspm*rate)
        reaction=Vec3(0.,load,0.)
        force=force+reaction
        torque=torque+arm.cross(reaction)
    conjugate=Quaternion(q.w,-q.x,-q.y,-q.z)
    omega=conjugate.rotate(vehicle.angular_velocity)
    momentum=Vec3(s.pitch_inertia_kg_m2*omega.x,1800.*omega.y,s.roll_inertia_kg_m2*omega.z)
    rhs=conjugate.rotate(torque)-omega.cross(momentum)
    expected_angular=q.rotate(Vec3(rhs.x/s.pitch_inertia_kg_m2,rhs.y/1800.,rhs.z/s.roll_inertia_kg_m2))
    policy=SpatialContactPolicy(LongitudinalContactPolicy(0.,100000.),1800.,0.)
    end=run(vehicle,f,s,VehicleControl(0.,0.,0.),dt=1e-7,policy=policy).suspension.vehicle
    assert ((end.velocity-vehicle.velocity)/1e-7-force/s.mass_kg).magnitude<2e-5
    assert ((end.angular_velocity-vehicle.angular_velocity)/1e-7-expected_angular).magnitude<2e-5


def test_work_refinement_exhaustion_rejects_then_refines_same_tolerance():
    from src.sandbox.independent_suspension_dynamics import SuspensionResolutionError
    f,_,_,s=ramp_translation()
    s=replace(s,integration_substeps=1)
    v=replace(f.initial_state.vehicle,position=f.initial_state.vehicle.position+Vec3(0.,.01,0.))
    p=SpatialContactPolicy(LongitudinalContactPolicy(.8,100000.,max_refinements=0),1800.,12000.)
    with pytest.raises(SuspensionResolutionError,match='refinement limit'):
        run(v,f,s,VehicleControl(0.,.5,0.),dt=.1,policy=p)
    r=run(v,f,s,VehicleControl(0.,.5,0.),dt=.1,policy=replace(p,traction=replace(p.traction,max_refinements=20)))
    assert len(r.audit.substeps)>1
    assert all(abs(a.energy_residual_j)<=a.energy_tolerance_j for a in r.audit.substeps)


@pytest.mark.parametrize('field,value',[
    ('yaw_inertia_kg_m2',0.),('yaw_inertia_kg_m2',float('inf')),
    ('lateral_damping_nspm',-1.),('lateral_damping_nspm',True),
    ('max_angular_speed_radps',0.),('max_angular_speed_radps',21.),('traction',None),
])
def test_spatial_policy_rejects_unregistered_numeric_domains(field,value):
    p=SpatialContactPolicy(LongitudinalContactPolicy(.8,100000.),1800.,12000.)
    with pytest.raises(ValueError):
        replace(p,**{field:value})


def test_world_height_origin_cannot_loosen_spatial_energy_tolerance():
    f,_,_,s=ramp_translation()
    first=run(f.initial_state.vehicle,f,s,VehicleControl(.2,.5,0.))
    shift=Vec3(0.,100.,0.)
    vehicle=replace(f.initial_state.vehicle,position=f.initial_state.vehicle.position+shift)
    surfaces=tuple(replace(p,origin=p.origin+shift) for p in f.track.surfaces)
    second=integrate_spatial_contact(vehicle,s,f.physics_config.body_geometry,surfaces,f.vehicle_parameters,
        VehicleControl(.2,.5,0.),SpatialContactPolicy(LongitudinalContactPolicy(.8,100000.),1800.,12000.),dt=.01)
    assert (first.suspension.vehicle.velocity-second.suspension.vehicle.velocity).magnitude<1e-10
    assert [a.energy_tolerance_j for a in first.audit.substeps]==pytest.approx(
        [a.energy_tolerance_j for a in second.audit.substeps],rel=1e-10)
