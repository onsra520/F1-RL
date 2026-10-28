"""Synthetic WP7 manifold assertions, no TRAIN/HELD-OUT data access."""
from dataclasses import replace
import pytest
from src.sandbox.contact_manifold import (
    ContactManifoldPolicy, WheelManifoldContact, ManifoldResolutionError,
    solve_wheel_contact_manifold,
)
from src.sandbox.geometry.vector import Vec3
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.independent_suspension_dynamics import IndependentSuspensionDynamicsPolicy
from src.sandbox.multi_surface_support import MultiSurfaceSuspensionPolicy
from src.sandbox.ramp_support import WheelContactPoints
from src.sandbox.vehicle import VehicleState

W=WheelContactPoints(
    Vec3(-.4,-.3,.8),Vec3(.4,-.3,.8),
    Vec3(-.4,-.3,-.8),Vec3(.4,-.3,-.8),
)
PHYSICS=IndependentSuspensionDynamicsPolicy(
    MultiSurfaceSuspensionPolicy(W,.12,.12),1200.,1800.,600.,
    30000.,2000.,150.,integration_substeps=8,
)
POLICY=ContactManifoldPolicy(yaw_inertia_kg_m2=1600.)
UP=Vec3(0.,1.,0.)

def vehicle(*, vx=0.,vy=-2., wz=0.,yaw=0.):
    q=Quaternion.from_axis_angle(Vec3(0.,1.,0.),yaw)
    return VehicleState(
        Vec3(0.,.3,0.),Vec3(vx,vy,0.),q,
        Vec3(0.,0.,wz),False,
    )

def contact(i, n=UP):
    return WheelManifoldContact(i,"flat",W.positions[i],n)

def run(state=None,contacts=None,policy=POLICY):
    return solve_wheel_contact_manifold(
        state or vehicle(),
        contacts if contacts is not None else tuple(contact(i) for i in range(4)),
        PHYSICS,policy,
    )

def test_symmetric_four_wheel_inelastic_has_zero_heave_and_spin_and_energy_loss():
    after,audit=run()
    assert after.velocity.y==pytest.approx(0.,abs=1e-5)
    assert after.angular_velocity.magnitude==pytest.approx(0.,abs=1e-5)
    assert audit.post_kinetic_energy_j < audit.pre_kinetic_energy_j
    assert len(audit.contacts)==4
    assert all(c.normal_impulse_ns>=0. for c in audit.contacts)
    assert sum(c.normal_impulse_ns for c in audit.contacts)==pytest.approx(2400.,abs=1e-3)
    assert audit.total_linear_impulse_ns.y==pytest.approx(2400.,abs=1e-3)
    assert audit.total_angular_impulse_nms.magnitude==pytest.approx(0.,abs=1e-3)

def test_two_wheel_manifold_handles_pitch_recovery_deterministically():
    state=vehicle(wz=0.)
    left,right=contact(0),contact(1)
    a,audit=run(state,(left,right))
    b,budit=run(state,(right,left))
    assert a==b
    assert audit==budit
    assert all(c.normal_impulse_ns>=0. for c in audit.contacts)
    assert audit.post_kinetic_energy_j<=audit.pre_kinetic_energy_j+POLICY.energy_tolerance_j

def test_single_oblique_friction_produces_yaw_torque_and_reduces_tangent_velocity():
    state=vehicle(vx=4.)
    friction=replace(POLICY,friction_coefficient=.65)
    after,audit=run(state,(contact(0),),friction)
    assert abs(after.angular_velocity.y)>0.
    assert audit.contacts[0].tangential_impulse_ns.magnitude>0.
    assert audit.contacts[0].tangential_impulse_ns.magnitude<=.65*audit.contacts[0].normal_impulse_ns+1e-7
    assert abs(after.velocity.x)<4.
    assert audit.post_kinetic_energy_j<=audit.pre_kinetic_energy_j+friction.energy_tolerance_j

def test_registered_yaw_inertia_changes_post_impact_spin_without_guessing():
    initial=vehicle(vx=4.)
    a,_=run(initial,(contact(0),),replace(POLICY,friction_coefficient=.65,yaw_inertia_kg_m2=700.))
    b,_=run(initial,(contact(0),),replace(POLICY,friction_coefficient=.65,yaw_inertia_kg_m2=5000.))
    assert abs(a.angular_velocity.y)>abs(b.angular_velocity.y)

def test_four_wheel_coulomb_friction_suppresses_horizontal_velocity_without_energy_gain():
    after,audit=run(vehicle(vx=3.),policy=replace(POLICY,friction_coefficient=.8))
    assert after.velocity.x < 3.
    assert audit.post_kinetic_energy_j<audit.pre_kinetic_energy_j
    assert all(c.tangential_impulse_ns.magnitude<=.8*c.normal_impulse_ns+1e-7 for c in audit.contacts)

def test_restitution_is_physical_bounce_not_fake_grounded():
    policy=replace(POLICY,restitution=.5)
    after,audit=run(policy=policy)
    assert not after.grounded
    assert after.velocity.y==pytest.approx(1.,abs=1e-5)
    assert audit.post_kinetic_energy_j < audit.pre_kinetic_energy_j
    assert all(c.outgoing_normal_speed_mps>0. for c in audit.contacts)

def test_perfectly_elastic_symmetric_impact_energy_is_conserved():
    policy=replace(POLICY,restitution=1.)
    after,audit=run(policy=policy)
    assert after.velocity.y==pytest.approx(2.,abs=1e-5)
    assert audit.post_kinetic_energy_j==pytest.approx(audit.pre_kinetic_energy_j,abs=1e-4)

def test_fails_closed_when_no_closing_wheel_or_no_convergence_budget():
    with pytest.raises(ManifoldResolutionError,match="no approaching"):
        run(vehicle(vy=2.))
    # A symmetric manifold can legitimately converge in one sweep. Exercise
    # the non-converged case with coupled sliding/friction and angular motion.
    difficult=vehicle(vx=4.,wz=.45)
    with pytest.raises(ManifoldResolutionError):
        run(difficult,policy=replace(
            POLICY,sweeps=1,friction_coefficient=.9,
            velocity_tolerance_mps=1e-12,
        ))

def test_duplicate_and_unknown_contact_geometry_rejected():
    with pytest.raises(ValueError,match="duplicate"):
        run(contacts=(contact(0),contact(0)))
    with pytest.raises(ValueError,match="geometry"):
        run(contacts=(WheelManifoldContact(0,"flat",Vec3(0.,-.3,.8),UP),))

@pytest.mark.parametrize("key,value",[
    ("yaw_inertia_kg_m2",0.),("yaw_inertia_kg_m2",True),
    ("friction_coefficient",-1.),("restitution",1.1),
    ("sweeps",0),("simultaneous_gap_m",float("inf")),
])
def test_invalid_policy_rejected(key,value):
    with pytest.raises(ValueError):
        replace(POLICY,**{key:value})
