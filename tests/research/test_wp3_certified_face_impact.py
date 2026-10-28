"""Face-impact certificate: geometry -> TOI -> normal impulse -> energy."""
from dataclasses import replace
import pytest
from src.simulation.certified_face_impact import certify_linear_face_impact
from src.simulation.body_collision import VehicleBodyGeometry
from src.simulation.geometry.vector import Vec3
from src.simulation.geometry.rotation import Quaternion
from src.simulation.track.volume import OrientedBoxVolume
from src.simulation.rigid_body_impact_response import RigidBodyImpactMass
from src.simulation.vehicle import VehicleState

Q=Quaternion.identity()
Z=Vec3(0.,0.,0.)
HULL=VehicleBodyGeometry(.4,.4,.4,Z)
WALL=OrientedBoxVolume(Z,Vec3(.1,2.,2.),Q)
MASS=RigidBodyImpactMass(1000.,Vec3(500.,500.,500.))

def state(x,vx):
    return VehicleState(Vec3(x,0.,0.),Vec3(vx,0.,0.),Q,Z,True)

def test_first_face_contact_has_bounded_toi_impulse_energy_and_no_tunneling():
    certificate=certify_linear_face_impact(
        state(.6,-2.),state(.0,-2.),HULL,WALL,MASS,duration_s=.3)
    assert certificate.schema_version=="wp3_face_impact_diagnostic_v1"
    assert .1499<certificate.first_certified_clear_s<certificate.first_witnessed_overlap_s<.1501
    assert certificate.certified_contact_time_uncertainty_s<=1e-6
    assert certificate.obstacle_face_normal==Vec3(1.,0.,0.)
    assert certificate.response.impulse_n_s.x>0.
    assert certificate.response.energy_lost_j>0.
    assert certificate.response.after_kinetic_j<=certificate.response.before_kinetic_j
    assert certificate.response.post_relative_normal_mps==pytest.approx(0.,abs=1e-6)

def test_still_separated_no_certificate():
    with pytest.raises(ValueError,match="no witnessed"):
        certify_linear_face_impact(state(.6,-2.),state(.5,-2.),
                                   HULL,WALL,MASS,duration_s=.05)

def test_head_on_contact_cannot_be_reused_with_receding_velocity():
    with pytest.raises(ValueError,match="ambiguous/non-face"):
        certify_linear_face_impact(state(.6,2.),state(.0,2.),
                                   HULL,WALL,MASS,duration_s=.3)

def test_contact_outside_obstacle_face_is_uncertified():
    far=OrientedBoxVolume(Z,Vec3(.1,.05,.05),Q)
    with pytest.raises(ValueError):
        certify_linear_face_impact(state(.6,-2.),state(.0,-2.),
                                   HULL,far,MASS,duration_s=.3)

def test_corner_grazing_remains_blocked():
    shifted=OrientedBoxVolume(Vec3(0.,.2,.2),Vec3(.1,.2,.2),Q)
    with pytest.raises(ValueError,match="ambiguous|no witnessed"):
        certify_linear_face_impact(state(.6,-2.),state(.0,-2.),
                                   HULL,shifted,MASS,duration_s=.3)
