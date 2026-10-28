"""Synthetic frictionless single-wheel impulse invariants; no scientific tracks."""
from dataclasses import replace
import pytest

from src.simulation.angular_landing_impulse import (
    AngularLandingImpulsePolicy, ImpulseResolutionError,
    solve_one_wheel_normal_impulse,
)
from src.simulation.geometry.vector import Vec3
from src.simulation.geometry.rotation import Quaternion
from src.simulation.independent_suspension_dynamics import IndependentSuspensionDynamicsPolicy
from src.simulation.multi_surface_support import MultiSurfaceSuspensionPolicy
from src.simulation.ramp_support import WheelContactPoints
from src.simulation.vehicle import VehicleState

WHEELS=WheelContactPoints(
    Vec3(-.4,-.3,.8),Vec3(.4,-.3,.8),
    Vec3(-.4,-.3,-.8),Vec3(.4,-.3,-.8),
)
PHYSICS=IndependentSuspensionDynamicsPolicy(
    MultiSurfaceSuspensionPolicy(WHEELS,.12,.12),
    1200.,1800.,600.,30000.,2000.,150.,
)
POLICY=AngularLandingImpulsePolicy()
UP=Vec3(0.,1.,0.)


def impact(*, vy=-2., omega=None, pitch=0., roll=0.):
    q=Quaternion.from_axis_angle(Vec3(1.,0.,0.),pitch)*Quaternion.from_axis_angle(Vec3(0.,0.,1.),roll)
    return VehicleState(Vec3(0.,.31,0.),Vec3(3.,vy,0.),q,
                        omega or Vec3(0.,0.,0.),False)


def solve(source, wheel=0, policy=POLICY):
    return solve_one_wheel_normal_impulse(
        source,WHEELS.positions[wheel],wheel,"road",UP,PHYSICS,policy,
    )


def test_normal_impulse_is_nonnegative_and_exact_restitution():
    source=impact()
    result,audit=solve(source)
    assert audit.magnitude_ns>0.
    assert audit.normal_speed_before_mps<0.
    assert audit.normal_speed_after_mps==pytest.approx(0.,abs=1e-9)
    assert audit.post_kinetic_energy_j < audit.pre_kinetic_energy_j
    assert result.velocity.x==source.velocity.x
    assert result.velocity.z==source.velocity.z
    assert result.position==source.position
    assert result.orientation==source.orientation
    assert not result.grounded


def test_unequal_left_right_angular_impulse_changes_pitch_and_roll():
    source=impact(vy=-1.5,omega=Vec3(.1,0.,0.))
    result,audit=solve(source,wheel=0)
    assert abs(audit.angular_delta_radps.x)>0.
    assert abs(audit.angular_delta_radps.z)>0.
    assert result.angular_velocity != source.angular_velocity
    assert audit.post_kinetic_energy_j < audit.pre_kinetic_energy_j
    assert audit.normal_speed_after_mps==pytest.approx(0.,abs=1e-9)


def test_restitution_bounds_kinetic_energy_and_relative_speed():
    source=impact(vy=-2.)
    after,audit=solve(source,policy=replace(POLICY,restitution=.5))
    assert audit.normal_speed_after_mps==pytest.approx(-.5*audit.normal_speed_before_mps)
    assert audit.pre_kinetic_energy_j + 1e-7 >= audit.post_kinetic_energy_j
    elastic,eaudit=solve(source,policy=replace(POLICY,restitution=1.,max_post_angular_speed_radps=10.))
    assert eaudit.pre_kinetic_energy_j==pytest.approx(eaudit.post_kinetic_energy_j,abs=1e-7)


def test_zero_or_upward_closing_speed_refused():
    with pytest.raises(ImpulseResolutionError,match="not approaching"):
        solve(impact(vy=1.))


def test_unsupported_yaw_spin_refused_instead_of_guessing_yaw_inertia():
    with pytest.raises(ImpulseResolutionError,match="yaw spin"):
        solve(impact(omega=Vec3(0.,.3,0.)))


def test_sloped_normal_refused_without_yaw_inertia():
    source=impact()
    normal=Vec3(0.,.995,.0998749218).normalized()
    with pytest.raises(ImpulseResolutionError,match="sloped angular impulse"):
        solve_one_wheel_normal_impulse(
            source,WHEELS.positions[0],0,"ramp",normal,PHYSICS,POLICY,
        )


def test_spin_and_restitution_input_validation():
    for value in (-1.,1.01,float("nan"),True):
        with pytest.raises(ValueError):
            replace(POLICY,restitution=value)
    for value in (0.,float("inf"),True):
        with pytest.raises(ValueError):
            replace(POLICY,max_pre_angular_speed_radps=value)


def test_identical_impact_is_deterministic():
    source=impact(vy=-1.5,omega=Vec3(.1,0.,0.))
    assert solve(source)==solve(source)
