"""Physics invariants for unforced 6-DOF angular motion on synthetic bodies."""
from dataclasses import replace
from math import cos,sin

import pytest

from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.torque_free_rotation import (
    AngularRigidBodyState, TorqueFreeRotationPolicy,
    RigidBodyIntegrationError, advance_torque_free_rotation,
)

POLICY=TorqueFreeRotationPolicy(1800.,1600.,600.,integration_steps=64)


def angular(w=Vec3(.1,.3,.2),rotation=None):
    return AngularRigidBodyState(rotation or Quaternion.identity(),w)


def test_no_spin_does_not_rotate_or_create_energy():
    initial=angular(Vec3(0.,0.,0.))
    last,audit=advance_torque_free_rotation(initial,POLICY,.5)
    assert last==initial
    assert audit.initial_energy_j==0.
    assert audit.final_energy_j==0.
    assert audit.final_world_angular_momentum==Vec3(0.,0.,0.)


def test_principal_axis_spin_is_uniform_with_exact_energy():
    policy=TorqueFreeRotationPolicy(10.,12.,14.,integration_steps=96)
    initial=angular(Vec3(0.,1.,0.))
    last,audit=advance_torque_free_rotation(initial,policy,.5)
    expected=Quaternion.from_axis_angle(Vec3(0.,1.,0.),.5)
    assert last.orientation.w==pytest.approx(expected.w,abs=1e-10)
    assert last.orientation.y==pytest.approx(expected.y,abs=1e-10)
    assert last.world_angular_velocity_radps==pytest.approx(initial.world_angular_velocity_radps)
    assert audit.final_energy_j==pytest.approx(audit.initial_energy_j,rel=1e-10)


def test_asymmetric_body_precession_preserves_energy_and_world_momentum():
    initial=angular()
    last,audit=advance_torque_free_rotation(initial,POLICY,.35)
    assert last.world_angular_velocity_radps!=initial.world_angular_velocity_radps
    assert audit.final_energy_j==pytest.approx(audit.initial_energy_j,rel=1e-8)
    assert (audit.final_world_angular_momentum-audit.initial_world_angular_momentum).magnitude<1e-5
    assert last.orientation!=initial.orientation


def test_world_momentum_audit_under_initial_body_yaw_rotation():
    q=Quaternion.from_axis_angle(Vec3(0.,1.,0.),.5)
    initial=angular(Vec3(.2,.15,.13),q)
    _,audit=advance_torque_free_rotation(initial,POLICY,.2)
    assert (audit.final_world_angular_momentum-audit.initial_world_angular_momentum).magnitude<1e-5


def test_substepping_and_deterministic_replay():
    initial=angular()
    a=advance_torque_free_rotation(initial,POLICY,.2)
    b=advance_torque_free_rotation(initial,POLICY,.2)
    assert a==b
    half, _=advance_torque_free_rotation(initial,POLICY,.1)
    end, _=advance_torque_free_rotation(half,POLICY,.1)
    assert abs(a[0].orientation.w-end.orientation.w)<1e-7
    assert (a[0].world_angular_velocity_radps-end.world_angular_velocity_radps).magnitude<1e-7


def test_limited_step_budget_fails_conservation_instead_of_claiming_valid():
    policy=TorqueFreeRotationPolicy(
        2000.,1200.,300.,integration_steps=1,
        max_angular_speed_radps=20.,
        energy_relative_tolerance=1e-14,
        momentum_relative_tolerance=1e-14,
    )
    with pytest.raises(RigidBodyIntegrationError,match="conservation"):
        advance_torque_free_rotation(angular(Vec3(5.,4.,3.)),policy,.5)


def test_spinning_outside_declared_domain_fails_closed():
    with pytest.raises(RigidBodyIntegrationError,match="initial angular speed"):
        advance_torque_free_rotation(
            angular(Vec3(0.,11.,0.)),POLICY,.1,
        )


@pytest.mark.parametrize("field,value",[
    ("pitch_inertia_kg_m2",0.),
    ("yaw_inertia_kg_m2",float("nan")),
    ("roll_inertia_kg_m2",True),
    ("integration_steps",0),
    ("integration_steps",1025),
    ("max_angular_speed_radps",-1.),
])
def test_invalid_policy_rejected(field,value):
    with pytest.raises(ValueError):
        replace(POLICY,**{field:value})


def test_invalid_time_rejected_and_zero_time_identity():
    initial=angular()
    assert advance_torque_free_rotation(initial,POLICY,0.)[0]==initial
    for duration in (-.1,float("inf"),True):
        with pytest.raises(ValueError):
            advance_torque_free_rotation(initial,POLICY,duration)
