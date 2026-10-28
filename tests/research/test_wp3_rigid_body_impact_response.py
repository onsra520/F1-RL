"""Rigid-body finite impulse physics: energy/work and fail-closed inputs."""
from dataclasses import replace
import pytest

from src.sandbox.body_collision import VehicleBodyGeometry
from src.sandbox.geometry.vector import Vec3
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.rigid_body_impact_response import RigidBodyImpactMass,rigid_body_normal_impact
from src.sandbox.vehicle import VehicleState

ZERO=Vec3(0.,0.,0.)
BODY=VehicleBodyGeometry(2.,2.,2.,ZERO)
MASS=RigidBodyImpactMass(1000.,Vec3(500.,500.,500.))
N=Vec3(1.,0.,0.)

def approaching(vx=-2.,vz=0.):
    return VehicleState(Vec3(2.,0.,0.),Vec3(vx,0.,vz),
                        Quaternion.identity(),ZERO,False)


def test_inelastic_center_face_hit_conserves_impulse_and_loses_energy():
    s=approaching()
    r=rigid_body_normal_impact(s,BODY,MASS,contact_point_world=Vec3(1.,0.,0.),
                               normal_obstacle_to_vehicle=N,restitution=0.)
    assert r.impulse_n_s.x==pytest.approx(2000.)
    assert r.angular_impulse_nms==ZERO
    assert r.velocity_after_mps==ZERO
    assert r.before_kinetic_j==pytest.approx(2000.)
    assert r.after_kinetic_j==pytest.approx(0.)
    assert r.energy_lost_j==pytest.approx(2000.)
    assert r.post_relative_normal_mps==pytest.approx(0.)


def test_elastic_center_face_hit_reverses_velocity_without_energy_gain():
    s=approaching()
    r=rigid_body_normal_impact(s,BODY,MASS,contact_point_world=Vec3(1.,0.,0.),
                               normal_obstacle_to_vehicle=N,restitution=1.)
    assert r.velocity_after_mps.x==pytest.approx(2.)
    assert r.energy_lost_j==pytest.approx(0.)
    assert r.post_relative_normal_mps==pytest.approx(2.)


def test_off_center_glancing_hit_has_angular_impulse_and_bounded_energy():
    s=approaching(vx=-3.,vz=1.)
    contact=Vec3(1.,.5,0.)
    r=rigid_body_normal_impact(s,BODY,MASS,contact_point_world=contact,
                               normal_obstacle_to_vehicle=N,restitution=.25)
    assert r.angular_impulse_nms.z<0.
    assert r.angular_velocity_after_radps.z<0.
    assert r.after_kinetic_j<=r.before_kinetic_j
    assert r.energy_lost_j>=0.
    assert r.post_relative_normal_mps==pytest.approx(-.25*r.pre_relative_normal_mps)
    assert r.velocity_after_mps.z==pytest.approx(1.)


def test_high_speed_contact_keeps_energy_audited():
    s=approaching(vx=-100.)
    r=rigid_body_normal_impact(s,BODY,MASS,contact_point_world=Vec3(1.,.8,0.),
                               normal_obstacle_to_vehicle=N,restitution=0.)
    assert r.after_kinetic_j<r.before_kinetic_j
    assert r.impulse_n_s.x>0.
    assert r.angular_impulse_nms.magnitude>0.


def test_separating_or_tangential_contact_is_rejected():
    for vx in (0.,1.):
        with pytest.raises(ValueError,match="separating or tangential"):
            rigid_body_normal_impact(approaching(vx=vx),BODY,MASS,
                contact_point_world=Vec3(1.,0.,0.),normal_obstacle_to_vehicle=N)


def test_interior_contact_is_not_a_certified_hull_surface():
    with pytest.raises(ValueError,match="surface"):
        rigid_body_normal_impact(approaching(),BODY,MASS,
            contact_point_world=Vec3(2.,0.,0.),normal_obstacle_to_vehicle=N)


@pytest.mark.parametrize("normal",[ZERO,Vec3(2.,0.,0.)])
def test_bad_normal_rejected(normal):
    with pytest.raises(ValueError,match="unit"):
        rigid_body_normal_impact(approaching(),BODY,MASS,
            contact_point_world=Vec3(1.,0.,0.),normal_obstacle_to_vehicle=normal)


@pytest.mark.parametrize("restitution",[-.1,1.01,float("nan"),float("inf"),True])
def test_invalid_restitution_fails_closed(restitution):
    with pytest.raises(ValueError,match="restitution"):
        rigid_body_normal_impact(approaching(),BODY,MASS,
            contact_point_world=Vec3(1.,0.,0.),
            normal_obstacle_to_vehicle=N,restitution=restitution)


def test_invalid_inertia_triangle_rejected():
    with pytest.raises(ValueError,match="triangle"):
        RigidBodyImpactMass(1000.,Vec3(100.,100.,1000.))


def test_collision_response_does_not_mutate_input_vehicle_state():
    v=approaching()
    before=repr(v)
    rigid_body_normal_impact(v,BODY,MASS,contact_point_world=Vec3(1.,0.,0.),
                             normal_obstacle_to_vehicle=N)
    assert repr(v)==before
