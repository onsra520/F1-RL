"""Finite road patch first-contact -> rigid normal impulse, no prior overlap."""
from dataclasses import replace
import pytest
from src.sandbox.accepted_road_body_contact import apply_first_road_body_contact
from src.sandbox.body_collision import VehicleBodyGeometry
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.geometry.vector import Vec3
from src.sandbox.rigid_body_impact_response import RigidBodyImpactMass
from src.sandbox.multi_surface_support import _atlas
from src.sandbox.track.patch import PlanarSurfacePatch
from src.sandbox.vehicle import VehicleState

Q=Quaternion.identity()
Z=Vec3(0.,0.,0.)
UP=Vec3(0.,1.,0.)
ROAD=PlanarSurfacePatch("road",Z,Vec3(0.,0.,1.),UP,10.,10.)
ATLAS=_atlas((ROAD,),1e-6)
BODY=VehicleBodyGeometry(.2,.2,.2,Z)
MASS=RigidBodyImpactMass(1000.,Vec3(500.,500.,500.))

def s(y,velocity=-2.):
    return VehicleState(Vec3(0.,y,0.),Vec3(0.,velocity,0.),Q,Z,False)

def test_first_witnessed_road_body_contact_applies_bounded_upward_impulse():
    result=apply_first_road_body_contact(s(.6),s(-.4),BODY,ATLAS,MASS,duration_s=.5)
    assert result is not None
    assert result.patch_id=="road"
    assert result.normal==UP
    assert result.contact.kind=="bounded_road_body"
    assert .2499<result.contact.first_clear_s<result.contact.witnessed_contact_s<.2501
    assert result.contact.postimpact.position==result.contact.preimpact.position
    assert result.contact.postimpact.velocity.y>result.contact.preimpact.velocity.y
    assert result.contact.impulse_n_s.y>0.
    assert result.contact.kinetic_after_j<result.contact.kinetic_before_j
    assert result.contact.energy_lost_j>0.

def test_body_road_contact_not_fabricated_outside_bounded_patch():
    far=PlanarSurfacePatch("far",Vec3(100.,0.,0.),Vec3(0.,0.,1.),UP,5.,5.)
    assert apply_first_road_body_contact(s(.6),s(-.4),BODY,_atlas((far,),1e-6),MASS,
                                         duration_s=.5) is None

def test_already_overlapping_chassis_cannot_claim_certified_first_contact():
    with pytest.raises(ValueError,match="already|overlapping"):
        apply_first_road_body_contact(s(0.),s(-.4),BODY,ATLAS,MASS,duration_s=.2)

def test_receding_contact_cannot_make_positive_collision_impulse():
    with pytest.raises(ValueError):
        apply_first_road_body_contact(s(.6,2.),s(-.4,2.),BODY,ATLAS,MASS,duration_s=.5)
