"""Exact frozen-seed WP3 edge-on-edge contact and fail-closed negatives."""
from dataclasses import replace
import pytest

from src.simulation.certified_edge_impact import (
    certify_unique_linear_edge_impact,apply_certified_edge_impulse)
from src.simulation.accepted_contact_transition import _apply_mass_impulse
from src.simulation.body_collision import VehicleBodyGeometry
from src.simulation.geometry.rotation import Quaternion
from src.simulation.geometry.vector import Vec3
from src.simulation.rigid_body_impact_response import RigidBodyImpactMass
from src.simulation.vehicle import VehicleState
from src.simulation.track.volume import OrientedBoxVolume

def fixture():
    start=VehicleState(
        Vec3(1.8448484421489493,0.5013823412347344,49.07814615765715),
        Vec3(1.4299844365690186,0.006916732122944183,3.1257735083550995),
        Quaternion(0.9509956060026251,-0.006879925149115927,
                   0.3083879612279628,-0.02137497048660089),
        Vec3(-0.025963978580060387,-0.5347520403539193,-0.10046355118180891),True)
    end=VehicleState(
        Vec3(1.8493191869507564,0.501403518788655,49.087931766131035),
        Vec3(1.4312009311356986,0.006629405737164679,3.137055548716254),
        Quaternion(0.9512491621679294,-0.006854227170377925,
                   0.3075944447214561,-0.021533894705983544),
        Vec3(-0.0250197999112583,-0.5333965201885347,-0.08932172583707712),True)
    body=VehicleBodyGeometry(1.8,1.2,4.,Vec3(0.,.3,0.))
    obstacle=OrientedBoxVolume(
        Vec3(3.5,1.,45.),Vec3(2.,1.,2.),Quaternion.identity())
    mass=RigidBodyImpactMass(1200.,Vec3(1800.,1800.,600.))
    return start,end,body,obstacle,mass

def test_exact_original_seed_unique_physical_edge_contact_and_energy():
    a,b,body,obstacle,mass=fixture()
    cert=certify_unique_linear_edge_impact(
        a,b,body,obstacle,mass,duration_s=.003125)
    assert cert.edge_indices==(2,1)
    assert 0.<cert.distance_m<1e-5
    assert .002013<cert.first_certified_clear_s<cert.first_witnessed_overlap_s<.002015
    assert cert.response.pre_relative_normal_mps < 0
    assert cert.response.post_relative_normal_mps >= -1e-7
    assert cert.response.energy_lost_j > 20.
    assert abs(cert.response.before_kinetic_j-cert.response.after_kinetic_j-
               cert.response.energy_lost_j)<1e-7
    impact=apply_certified_edge_impulse(
        a,b,body,obstacle,mass,duration_s=.003125)
    assert impact.kind=="obstacle_edge"
    assert impact.energy_lost_j==cert.response.energy_lost_j
    assert ((impact.postimpact.velocity-impact.preimpact.velocity)*1200-
            impact.impulse_n_s).magnitude<1e-7

def test_edge_certificate_does_not_invent_contact_on_separated_obstacle():
    a,b,body,obstacle,mass=fixture()
    moved=replace(obstacle,center=Vec3(35.,1.,45.))
    with pytest.raises(ValueError):
        certify_unique_linear_edge_impact(a,b,body,moved,mass,duration_s=.003125)

def test_certifier_rejects_unregistered_forged_axis_margin():
    a,b,body,obstacle,mass=fixture()
    with pytest.raises(ValueError,match="positive finite bounded"):
        certify_unique_linear_edge_impact(a,b,body,obstacle,mass,
            duration_s=.003125,minimum_axis_margin_m=.1)

def test_cannot_forge_impulse_energy_or_raise_contact_tolerance():
    a,b,body,obstacle,mass=fixture()
    with pytest.raises(ValueError):
        certify_unique_linear_edge_impact(a,b,body,obstacle,mass,
            duration_s=.003125,max_contact_distance_m=.1)
    good=apply_certified_edge_impulse(a,b,body,obstacle,mass,duration_s=.003125)
    with pytest.raises(ValueError,match="energy"):
        _apply_mass_impulse(good.preimpact,mass,
            good.impulse_n_s,good.angular_impulse_nms,good.kinetic_before_j,
            good.kinetic_after_j,good.energy_lost_j+100.,
            first_clear=good.first_clear_s,
            witness=good.witnessed_contact_s,kind="obstacle_edge")
