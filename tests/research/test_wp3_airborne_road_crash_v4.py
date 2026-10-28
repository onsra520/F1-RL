"""Physics v4: road contact while the wheels are airborne is NOT grounded."""
from dataclasses import replace
import pytest
from src.simulation.geometry.vector import Vec3
from src.simulation.geometry.rotation import Quaternion
from src.simulation.body_collision import VehicleBodyGeometry
from src.simulation.track.volume import OrientedBoxVolume
from src.simulation.track.patch import PlanarSurfacePatch
from src.simulation.multi_surface_support import _atlas
from src.simulation.vehicle import VehicleState
from src.simulation.rigid_body_impact_response import RigidBodyImpactMass
from src.simulation.center_trajectory import CenterSegment
from src.simulation.linear_first_body_contact import certify_first_linear_body_contact
from src.simulation.accepted_road_body_contact import apply_first_road_body_contact
from src.simulation.physical_crash import PhysicalCrash
from src.rl.crash_terminal_registered import SOURCES,descriptor

Q=Quaternion.identity()
Z=Vec3(0.,0.,0.)
BODY=VehicleBodyGeometry(1.,.2,1.,Z)
ROAD=PlanarSurfacePatch("road",Z,Vec3(0.,0.,1.),Vec3(0.,1.,0.),20.,20.)
ATLAS=_atlas((ROAD,),1e-6)
MASS=RigidBodyImpactMass(1000.,Vec3(500.,500.,500.))
H=.2
G=10.

def airborne_pair():
    a=VehicleState(Vec3(0.,.15,0.),Z,Q,Z,False)
    b=VehicleState(Vec3(0.,-.05,0.),Vec3(0.,-G*H,0.),Q,Z,False)
    return a,b

def witnessed():
    a,b=airborne_pair()
    return apply_first_road_body_contact(a,b,BODY,ATLAS,MASS,duration_s=H,
                                          ballistic_gravity_mps2=G)

def test_ballistic_contact_is_later_than_spurious_linear_chord():
    a,b=airborne_pair()
    thin=OrientedBoxVolume(Z,Vec3(10.,1e-9,10.),Q)
    linear=certify_first_linear_body_contact(a,b,BODY,thin,duration_s=H)
    true=certify_first_linear_body_contact(a,b,BODY,thin,duration_s=H,
                                           ballistic_gravity_mps2=G)
    assert linear.first_witnessed_overlap_s==pytest.approx(.05,abs=1e-6)
    assert true.first_witnessed_overlap_s==pytest.approx(.1,abs=1e-6)
    assert true.first_witnessed_overlap_s>linear.first_witnessed_overlap_s

def test_airborne_applied_impulse_and_exact_gravity_prefix():
    a,b=airborne_pair()
    w=witnessed()
    assert w is not None
    c=w.contact
    assert c.kind=="bounded_road_body"
    assert not c.preimpact.grounded and not c.postimpact.grounded
    assert c.witnessed_contact_s==pytest.approx(.1,abs=1e-6)
    ballistic_y=a.position.y+a.velocity.y*c.witnessed_contact_s-.5*G*c.witnessed_contact_s**2
    assert c.preimpact.position.y==pytest.approx(ballistic_y,abs=1e-12)
    assert c.preimpact.velocity.y==pytest.approx(-G*c.witnessed_contact_s,abs=1e-9)
    assert c.impulse_n_s.y>0.
    assert c.kinetic_after_j<=c.kinetic_before_j
    assert c.energy_lost_j==pytest.approx(c.kinetic_before_j-c.kinetic_after_j)
    prefix=(CenterSegment(0.,c.witnessed_contact_s,a.position,c.preimpact.position,
                          a.velocity,G),
            CenterSegment(c.witnessed_contact_s,H,c.preimpact.position,c.preimpact.position))
    event=PhysicalCrash("road-patch:road",c.witnessed_contact_s,c.first_clear_s,
                         c.witnessed_contact_s,0,c,prefix,
                         version="wp3_airborne_applied_road_crash_v4_dev")
    assert event.version in SOURCES
    assert SOURCES[event.version][1]=="airborne_road_obb_impulse_absorbing_v4_development"
    assert descriptor()["scientific_evidence_allowed"] is False
    with pytest.raises(ValueError):
        replace(event,version="wp3_grounded_applied_road_crash_v4_dev")
    with pytest.raises(ValueError):
        replace(event,obstacle_id="wheel-hard-stop:road")

def test_airborne_rejects_wrong_gravity_or_false_phase():
    a,b=airborne_pair()
    with pytest.raises(ValueError,match="exact gravity"):
        apply_first_road_body_contact(a,b,BODY,ATLAS,MASS,duration_s=H,
                                      ballistic_gravity_mps2=9.81)
    with pytest.raises(ValueError,match="exact gravity"):
        apply_first_road_body_contact(replace(a,grounded=True),b,BODY,ATLAS,MASS,
                                      duration_s=H,ballistic_gravity_mps2=G)
    with pytest.raises(ValueError,match="gravity"):
        apply_first_road_body_contact(a,b,BODY,ATLAS,MASS,duration_s=H,
                                      ballistic_gravity_mps2=float("nan"))

def test_grounded_source_does_not_accept_airborne_version():
    a,b=airborne_pair()
    assert "wp3_grounded_applied_road_crash_v4_dev" in SOURCES
    assert "wp3_airborne_applied_road_crash_v4_dev" in SOURCES
    assert SOURCES["wp3_grounded_applied_road_crash_v4_dev"]!=SOURCES[
        "wp3_airborne_applied_road_crash_v4_dev"]
